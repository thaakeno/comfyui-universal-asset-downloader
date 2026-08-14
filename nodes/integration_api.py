from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import web
from server import PromptServer

from . import smart_asset_service as service
from .hf_xet_download import configure_xet_environment, stage_huggingface_asset

UAD_VERSION = "2.1.3"
MAX_BATCH_ITEMS = 64

# Extend the v2 destination vocabulary before any analysis happens. These
# destinations are still constrained to ComfyUI/models by service.safe_target.
service.ALLOWED_DESTINATIONS.update({"vae_approx", "pdd_heads"})

_ORIGINAL_INFER_DESTINATION = service.infer_destination
_ORIGINAL_DOWNLOAD_ASSETS = service.download_assets


def _console(message: str) -> None:
    print(f"[UAD] {message}", flush=True)


def _human_rate(byte_count: int, seconds: float) -> str:
    if byte_count <= 0 or seconds <= 0:
        return ""
    return f"{service.human_size(int(byte_count / seconds))}/s"


def _enhanced_infer_destination(
    repo_id: str,
    filename: str,
    declared_type: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    text = f"{repo_id} {filename} {' '.join(tags or [])}".lower().replace("-", "_")

    if "pdd" in text and any(token in text for token in ("head", "heads", "displacement")):
        return {
            "asset_type": "PDD Heads",
            "destination": "pdd_heads",
            "confidence": 0.995,
            "reason": "MiniMax H3 PDD displacement-head signature",
        }
    if "pdd" in text and any(token in text for token in ("lora", "student", "adapter")):
        return {
            "asset_type": "PDD LoRA",
            "destination": "loras",
            "confidence": 0.995,
            "reason": "MiniMax H3 PDD student-adapter signature",
        }

    if any(token in text for token in ("taeh3", "vae_approx", "tiny_vae", "preview_vae")):
        return {
            "asset_type": "Preview VAE",
            "destination": "vae_approx",
            "confidence": 0.99,
            "reason": "preview/tiny VAE signature",
        }

    return _ORIGINAL_INFER_DESTINATION(repo_id, filename, declared_type, tags)


service.infer_destination = _enhanced_infer_destination


def _trusted_host_for_provider(provider: str, host: str) -> bool:
    provider = str(provider or "").strip().lower()
    host = str(host or "").strip().lower().split(":", 1)[0]
    if provider == "huggingface":
        return host in {"huggingface.co", "www.huggingface.co"}
    if provider == "civitai":
        return host in {"civitai.com", "www.civitai.com"}
    return False


def validate_install_asset(asset: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(asset, dict):
        raise ValueError("Installer item must be an object.")

    provider = str(asset.get("provider") or "").strip().lower()
    if provider not in {"huggingface", "civitai"}:
        raise ValueError(f"Smart install does not accept provider {provider or 'unknown'!r}.")

    download_url = str(asset.get("download_url") or "").strip()
    parsed = urlparse(download_url)
    if parsed.scheme != "https" or not _trusted_host_for_provider(provider, parsed.netloc):
        raise ValueError("Refusing an untrusted or non-HTTPS download URL.")

    destination = str(asset.get("destination") or "").strip().lower()
    filename = str(asset.get("filename") or "").strip()
    target = service.safe_target(destination, filename)

    if target.suffix.lower() not in service.MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported model file extension: {target.suffix or '<none>'}")

    size = int(asset.get("size_bytes") or 0)
    if size < 0:
        raise ValueError("Asset size cannot be negative.")

    normalized = {
        **asset,
        "provider": provider,
        "destination": destination,
        "filename": target.name,
        "download_url": download_url,
    }

    if provider == "huggingface" and (not normalized.get("repo_id") or not normalized.get("remote_path")):
        try:
            repo_id, revision, direct_file, _prefix = service._hf_parse(download_url)
            if direct_file:
                normalized["repo_id"] = repo_id
                normalized["revision"] = revision
                normalized["remote_path"] = direct_file
        except Exception:
            pass

    return normalized


def _download_huggingface_xet(
    asset: dict[str, Any],
    hf_token: str = "",
    force: bool = False,
    progress_callback=None,
) -> dict[str, Any]:
    target = service.safe_target(asset.get("destination") or "unclassified", asset.get("filename") or "")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(asset.get("size_bytes") or 0) or None
    expected_hash = str(asset.get("sha256") or "").lower()

    if target.exists() and not force:
        verify_started = time.monotonic()
        _console(f"  verify existing · {target.name}")
        verification = service.verify_file(target, expected_size, expected_hash)
        if verification["ok"]:
            _console(f"  ✓ existing file verified · {time.monotonic() - verify_started:.2f}s")
            return {**verification, "skipped": True, "asset": asset, "backend": "hf_xet"}
        raise ValueError(
            f"{target.name} already exists but failed verification. Enable force download to replace it safely."
        )

    if expected_size:
        free = shutil.disk_usage(target.parent).free
        required = expected_size + max(512 * 1024 * 1024, int(expected_size * 0.03))
        if free < required:
            raise OSError(
                f"Not enough free disk space for {target.name}: "
                f"need about {service.human_size(required)}, have {service.human_size(free)}."
            )

    downloaded = None
    stage_dir = None
    profile: dict[str, Any] = {}
    try:
        downloaded, stage_dir, profile = stage_huggingface_asset(
            asset,
            service.models_dir(),
            hf_token=hf_token,
            force=force,
            progress_callback=progress_callback,
        )
        _console(
            "  Xet transfer · "
            f"hp={'ON' if profile.get('high_performance') else 'adaptive'} · "
            f"hub={profile.get('huggingface_hub', 'unknown')} · hf_xet={profile.get('hf_xet', 'unknown')} · "
            f"progress={profile.get('progress_bridge', 'unknown')}"
        )

        verify_started = time.monotonic()
        if expected_hash:
            _console(f"  verify SHA256 · {service.human_size(expected_size)}")
        else:
            _console("  verify format + size")
        staged_verification = service.verify_file(downloaded, expected_size, expected_hash)
        if not staged_verification["ok"]:
            raise IOError(staged_verification.get("message") or f"Verification failed for {target.name}.")
        _console(f"  ✓ verified · {time.monotonic() - verify_started:.2f}s")

        # Same-filesystem staging means os.replace is atomic. The staged bytes
        # have already passed SHA256 verification, so do not reread a multi-GB
        # model a second time after the atomic rename. Recheck size/header only.
        os.replace(downloaded, target)
        verification = service.verify_file(target, expected_size, "")
        if not verification["ok"]:
            raise IOError(verification.get("message") or f"Final verification failed for {target.name}.")
        verification["sha256"] = staged_verification.get("sha256") or expected_hash
        verification["verification_level"] = "deep"
        return {
            **verification,
            "skipped": False,
            "asset": asset,
            "backend": "hf_xet",
            "xet": profile,
        }
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def secure_download_assets(
    assets: list[dict[str, Any]],
    hf_token: str = "",
    civitai_api_key: str = "",
    force: bool = False,
    progress_callback=None,
) -> list[dict[str, Any]]:
    if not isinstance(assets, list) or not assets:
        raise ValueError("Select at least one model file to install.")
    if len(assets) > MAX_BATCH_ITEMS:
        raise ValueError(f"Refusing to install more than {MAX_BATCH_ITEMS} files in one batch.")

    validated = [validate_install_asset(asset) for asset in assets]
    results: list[dict[str, Any]] = []
    for asset in validated:
        if asset.get("provider") == "huggingface" and asset.get("repo_id") and asset.get("remote_path"):
            results.append(
                _download_huggingface_xet(
                    asset,
                    hf_token=hf_token,
                    force=force,
                    progress_callback=progress_callback,
                )
            )
        else:
            results.extend(
                _ORIGINAL_DOWNLOAD_ASSETS(
                    [asset],
                    hf_token=hf_token,
                    civitai_api_key=civitai_api_key,
                    force=force,
                    progress_callback=progress_callback,
                )
            )
    return results


service.download_assets = secure_download_assets


def _models_root() -> str:
    return str(service.models_dir())


def _xet_profile() -> dict[str, Any]:
    try:
        return configure_xet_environment(service.models_dir())
    except Exception as exc:
        return {"backend": "huggingface_hub+hf_xet", "error": str(exc), "high_performance": False}


def _send_external_progress(
    node_id: str,
    status: str,
    progress: float | None = None,
    asset: dict[str, Any] | None = None,
    *,
    downloaded_bytes: int | None = None,
    total_bytes: int | None = None,
    file_index: int | None = None,
    file_count: int | None = None,
) -> None:
    if not node_id:
        return
    payload: dict[str, Any] = {"node": str(node_id), "status": status}
    if progress is not None:
        payload["progress"] = max(0.0, min(100.0, float(progress)))
    if asset:
        payload["filename"] = asset.get("filename", "")
        payload["destination"] = asset.get("destination", "")
    if downloaded_bytes is not None:
        payload["downloaded_bytes"] = max(0, int(downloaded_bytes))
    if total_bytes is not None:
        payload["total_bytes"] = max(0, int(total_bytes))
    if file_index is not None:
        payload["file_index"] = max(1, int(file_index))
    if file_count is not None:
        payload["file_count"] = max(1, int(file_count))
    PromptServer.instance.send_sync("uad-progress", payload)


@PromptServer.instance.routes.get("/uad/status")
async def api_status(_request):
    profile = _xet_profile()
    return web.json_response(
        {
            "ok": True,
            "name": "Universal Asset Downloader",
            "version": UAD_VERSION,
            "models_dir": _models_root(),
            "huggingface": profile,
            "capabilities": {
                "analyze": True,
                "verify": True,
                "verify_fast": True,
                "verify_fast_hashless": True,
                "install": True,
                "provider_hashes": True,
                "atomic_downloads": True,
                "external_integration": True,
                "nonblocking_analysis": True,
                "rich_progress": True,
                "console_progress": True,
                "console_progress_compact": True,
                "pdd_heads": True,
                "hf_xet": True,
                "hf_xet_high_performance": bool(profile.get("high_performance")),
            },
        },
        headers={"Cache-Control": "no-store"},
    )


@PromptServer.instance.routes.post("/uad/install")
async def api_install(request):
    node_id = ""
    try:
        payload = await request.json()
        items = payload.get("items") or []
        node_id = str(payload.get("node_id") or "")
        hf_token = str(payload.get("hf_token") or "")
        civitai_api_key = str(payload.get("civitai_api_key") or "")
        force = bool(payload.get("force", False))

        validated = [validate_install_asset(item) for item in items]
        total_expected = sum(int(item.get("size_bytes") or 0) for item in validated)
        completed_before = 0
        file_count = len(validated)
        profile = _xet_profile()
        install_started = time.monotonic()
        _console(
            f"── Install · {file_count} file(s) · {service.human_size(total_expected)} · "
            f"hf_xet · HP {'ON' if profile.get('high_performance') else 'adaptive'} ──"
        )

        results: list[dict[str, Any]] = []
        for file_zero_index, asset in enumerate(validated):
            file_index = file_zero_index + 1
            file_started = time.monotonic()
            console_state = {"percent": -10, "time": file_started, "done": False}

            def progress(downloaded: int, total: int, current_asset: dict[str, Any]) -> None:
                if total_expected > 0:
                    denominator = total_expected
                    current_total = total or int(current_asset.get("size_bytes") or 0)
                    overall = (completed_before + min(downloaded, current_total or downloaded)) / denominator * 100.0
                else:
                    file_fraction = (downloaded / total) if total else 0.0
                    overall = ((file_zero_index + file_fraction) / max(1, file_count)) * 100.0
                _send_external_progress(
                    node_id,
                    f"Downloading {current_asset.get('filename', 'model')}",
                    overall,
                    current_asset,
                    downloaded_bytes=downloaded,
                    total_bytes=total,
                    file_index=file_index,
                    file_count=file_count,
                )

                if downloaded <= 0:
                    return
                now = time.monotonic()
                elapsed = max(0.001, now - file_started)
                file_percent = (downloaded / total * 100.0) if total else 0.0
                complete = bool(total and downloaded >= total)
                if complete and console_state["done"]:
                    return
                should_print = complete or file_percent >= console_state["percent"] + 10 or now - console_state["time"] >= 1.5
                if not should_print:
                    return
                if complete:
                    console_state["done"] = True
                console_state["percent"] = int(file_percent // 10) * 10
                console_state["time"] = now
                amount = (
                    f"{service.human_size(downloaded)} / {service.human_size(total)}"
                    if total
                    else service.human_size(downloaded)
                )
                pct = f" · {min(100.0, file_percent):.0f}%" if total else ""
                rate = _human_rate(downloaded, elapsed)
                rate_note = f" · {rate}" if rate else ""
                _console(f"  [{file_index}/{file_count}] {amount}{pct}{rate_note}")

            starting_progress = (file_zero_index / max(1, file_count)) * 100.0
            backend = "Xet" if asset.get("provider") == "huggingface" else asset.get("provider", "provider")
            expected = int(asset.get("size_bytes") or 0)
            size_note = f" · {service.human_size(expected)}" if expected else ""
            _console(
                f"↓ [{file_index}/{file_count}] {asset.get('filename', 'model')}{size_note} · "
                f"{backend} → models/{asset.get('destination', 'unclassified')}"
            )
            _send_external_progress(
                node_id,
                f"Preparing {asset.get('filename', 'model')} · {backend}",
                starting_progress,
                asset,
                downloaded_bytes=0,
                total_bytes=expected,
                file_index=file_index,
                file_count=file_count,
            )
            batch = await asyncio.to_thread(
                secure_download_assets,
                [asset],
                hf_token,
                civitai_api_key,
                force,
                progress,
            )
            results.extend(batch)
            completed_before += expected
            result = batch[-1] if batch else {}
            _console(
                f"✓ [{file_index}/{file_count}] {asset.get('filename', 'model')} · "
                f"{'reused' if result.get('skipped') else 'installed'} · "
                f"{time.monotonic() - file_started:.2f}s"
            )

        _send_external_progress(
            node_id,
            "Install complete",
            100.0,
            downloaded_bytes=total_expected if total_expected else None,
            total_bytes=total_expected if total_expected else None,
            file_index=file_count if file_count else None,
            file_count=file_count if file_count else None,
        )
        _console(
            f"✓ Install complete · {file_count} file(s) · {service.human_size(total_expected)} · "
            f"{time.monotonic() - install_started:.2f}s"
        )
        compact = []
        for result in results:
            asset = result.get("asset") or {}
            compact.append(
                {
                    "ok": bool(result.get("ok")),
                    "status": result.get("status"),
                    "message": result.get("message"),
                    "path": result.get("path"),
                    "size_bytes": result.get("size_bytes"),
                    "sha256": result.get("sha256"),
                    "skipped": bool(result.get("skipped")),
                    "filename": asset.get("filename"),
                    "destination": asset.get("destination"),
                    "backend": result.get("backend"),
                    "xet": result.get("xet"),
                }
            )
        return web.json_response({"ok": True, "results": compact})
    except Exception as exc:
        _send_external_progress(node_id, f"Install failed: {exc}", 0.0)
        _console(f"✗ Install failed · {exc}")
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


__all__ = [
    "UAD_VERSION",
    "secure_download_assets",
    "validate_install_asset",
]
