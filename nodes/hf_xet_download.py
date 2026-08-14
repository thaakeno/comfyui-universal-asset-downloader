from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
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


def stage_huggingface_asset(
    asset: dict[str, Any],
    models_root: Path,
    hf_token: str = "",
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Download one HF file with hf_hub_download/hf_xet into same-FS staging.

    Returns `(downloaded_file, stage_dir, xet_profile)`. The caller owns final
    verification and atomic promotion into the destination path.
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

    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                revision=revision,
                local_dir=str(stage_dir),
                token=hf_token or None,
                force_download=force,
                tqdm_class=tqdm_class,
            )
        ).resolve()
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
