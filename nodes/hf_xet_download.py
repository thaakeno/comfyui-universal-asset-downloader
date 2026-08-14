from __future__ import annotations

import contextlib
import inspect
import os
import shutil
import tempfile
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm

ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def _truthy(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "high", "hp"}:
        return True
    if normalized in {"0", "false", "no", "off", "adaptive", "normal"}:
        return False
    return None


def _package_version(name: str) -> str:
    with contextlib.suppress(Exception):
        return metadata.version(name)
    return "unknown"


def total_memory_bytes() -> int:
    with contextlib.suppress(Exception):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and pages > 0:
            return page_size * pages
    return 0


def looks_like_lightning(models_root: Path) -> bool:
    """Use Lightning's persistent /teamspace mount as a high-bandwidth hint only."""
    with contextlib.suppress(Exception):
        return str(models_root.resolve()).startswith("/teamspace/")
    return False


def configure_xet_environment(models_root: Path) -> dict[str, Any]:
    """Configure a portable Hugging Face/Xet runtime without shelling out to `hf`.

    `huggingface_hub` automatically uses `hf_xet` when available. UAD respects
    explicit HF/UAD overrides, enables HP mode for the Lightning /teamspace
    profile used by H3 Studio, enables it on >=64 GiB machines elsewhere, and
    otherwise leaves Xet's adaptive concurrency controller in charge.
    """

    models_root = models_root.expanduser().resolve()
    comfy_root = models_root.parent
    cache_root = comfy_root / ".cache" / "huggingface"
    hub_cache = cache_root / "hub"
    xet_cache = cache_root / "xet"
    hub_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_XET_CACHE", str(xet_cache))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    explicit_uad = _truthy(os.environ.get("UAD_HF_XET_HIGH_PERFORMANCE"))
    explicit_hf = _truthy(os.environ.get("HF_XET_HIGH_PERFORMANCE"))
    memory = total_memory_bytes()

    if explicit_uad is not None:
        high_performance = explicit_uad
        reason = "UAD override"
        if high_performance:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        else:
            os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    elif explicit_hf is not None:
        high_performance = explicit_hf
        reason = "existing HF setting"
    elif looks_like_lightning(models_root):
        high_performance = True
        reason = "Lightning /teamspace profile"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    elif memory >= 64 * 1024**3:
        high_performance = True
        reason = ">=64 GiB RAM"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    else:
        high_performance = False
        reason = "adaptive concurrency"
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)

    return {
        "backend": "huggingface_hub+hf_xet",
        "huggingface_hub": _package_version("huggingface_hub"),
        "hf_xet": _package_version("hf_xet"),
        "high_performance": high_performance,
        "reason": reason,
        "memory_bytes": memory,
        "hf_home": os.environ.get("HF_HOME", ""),
        "hub_cache": os.environ.get("HF_HUB_CACHE", ""),
        "xet_cache": os.environ.get("HF_XET_CACHE", ""),
    }


def progress_tqdm(asset: dict[str, Any], callback: ProgressCallback | None, expected_size: int | None):
    class UadHubTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._uad_last = -1

        def _emit(self) -> None:
            if callback is None:
                return
            current = max(0, int(getattr(self, "n", 0) or 0))
            total = int(getattr(self, "total", 0) or expected_size or 0)
            if current == self._uad_last and current < total:
                return
            self._uad_last = current
            callback(current, total, asset)

        def update(self, n=1):
            changed = super().update(n)
            self._emit()
            return changed

        def close(self):
            self._emit()
            return super().close()

    return UadHubTqdm


def _materialized_size(path: Path) -> int:
    """Estimate bytes physically materialized, including sparse Xet reconstructions."""
    with contextlib.suppress(OSError):
        stat = path.stat()
        logical = max(0, int(stat.st_size))
        blocks = max(0, int(getattr(stat, "st_blocks", 0) or 0) * 512)
        if blocks and logical > blocks:
            return blocks
        return logical
    return 0


def _staging_progress_bytes(stage_dir: Path, expected_size: int | None) -> int:
    """Track the largest materializing file without counting Hub metadata twice."""
    best = 0
    with contextlib.suppress(OSError):
        for path in stage_dir.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name.endswith((".lock", ".json")) or name == ".gitignore":
                continue
            best = max(best, _materialized_size(path))
    if expected_size:
        return min(best, expected_size)
    return best


def _start_legacy_progress_monitor(
    stage_dir: Path,
    asset: dict[str, Any],
    expected_size: int | None,
    callback: ProgressCallback | None,
) -> tuple[threading.Event, threading.Thread | None]:
    """Bridge progress for pre-1.0 Hub APIs that lack `tqdm_class=`.

    Xet may reconstruct a sparse destination with its logical size visible early,
    so Linux `st_blocks` is used when available. On other platforms this falls
    back to logical file growth. This monitor never controls the transfer; it
    only reports what is physically appearing in UAD's same-filesystem staging.
    """

    stop = threading.Event()
    if callback is None:
        return stop, None

    def run() -> None:
        last = -1
        while not stop.wait(0.2):
            current = _staging_progress_bytes(stage_dir, expected_size)
            if current <= last:
                continue
            last = current
            callback(current, expected_size or 0, asset)

    thread = threading.Thread(target=run, name="uad-hf-xet-progress", daemon=True)
    thread.start()
    return stop, thread


def _download_kwargs(
    hf_hub_download,
    *,
    repo_id: str,
    remote_path: str,
    revision: str,
    stage_dir: Path,
    hf_token: str,
    force: bool,
    tqdm_class,
) -> tuple[dict[str, Any], bool]:
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": remote_path,
        "revision": revision,
        "local_dir": str(stage_dir),
        "token": hf_token or None,
        "force_download": force,
    }
    supports_tqdm_class = False
    with contextlib.suppress(Exception):
        supports_tqdm_class = "tqdm_class" in inspect.signature(hf_hub_download).parameters
    if supports_tqdm_class:
        kwargs["tqdm_class"] = tqdm_class
    return kwargs, supports_tqdm_class


def stage_huggingface_asset(
    asset: dict[str, Any],
    models_root: Path,
    hf_token: str = "",
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Download one HF file with hf_hub_download/hf_xet into same-FS staging.

    Compatible with the H3 Studio 0.36.x Hub stack as well as current 1.x Hub
    releases. New Hub versions use the direct tqdm callback. Older Hub versions
    keep full Xet acceleration and get UI progress from the staging monitor.
    """

    from huggingface_hub import hf_hub_download

    repo_id = str(asset.get("repo_id") or "").strip()
    remote_path = str(asset.get("remote_path") or "").strip()
    revision = str(asset.get("revision") or "main").strip() or "main"
    if not repo_id or not remote_path:
        raise ValueError("Hugging Face Xet download requires repo_id and remote_path metadata.")

    models_root = models_root.expanduser().resolve()
    profile = configure_xet_environment(models_root)
    stage_parent = models_root / ".uad_staging"
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix="hf-", dir=str(stage_parent)))
    expected_size = int(asset.get("size_bytes") or 0) or None
    tqdm_class = progress_tqdm(asset, progress_callback, expected_size)
    kwargs, supports_tqdm_class = _download_kwargs(
        hf_hub_download,
        repo_id=repo_id,
        remote_path=remote_path,
        revision=revision,
        stage_dir=stage_dir,
        hf_token=hf_token,
        force=force,
        tqdm_class=tqdm_class,
    )
    profile["progress_bridge"] = "hub-tqdm" if supports_tqdm_class else "staging-monitor"
    stop = threading.Event()
    monitor = None

    try:
        if not supports_tqdm_class:
            stop, monitor = _start_legacy_progress_monitor(stage_dir, asset, expected_size, progress_callback)
        downloaded = Path(hf_hub_download(**kwargs)).resolve()
        if not downloaded.is_file():
            raise IOError(f"Hugging Face did not produce the expected file: {downloaded}")
        downloaded.relative_to(models_root)
        if progress_callback:
            size = downloaded.stat().st_size
            progress_callback(size, expected_size or size, asset)
        return downloaded, stage_dir, profile
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=1.0)
