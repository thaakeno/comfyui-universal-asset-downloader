from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import web
from server import PromptServer

from . import smart_asset_service as service

UAD_VERSION = "2.0.1"
MAX_BATCH_ITEMS = 64

# Extend the v2 destination vocabulary before any analysis happens. These
# destinations are still constrained to ComfyUI/models by service.safe_target.
service.ALLOWED_DESTINATIONS.update({"vae_approx", "pdd_heads"})

_ORIGINAL_INFER_DESTINATION = service.infer_destination
_ORIGINAL_DOWNLOAD_ASSETS = service.download_assets


def _enhanced_infer_destination(
    repo_id: str,
    filename: str,
    declared_type: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    text = f"{repo_id} {filename} {' '.join(tags or [])}".lower().replace("-", "_")

    # MiniMax H3 PDD is a matched student-LoRA + displacement-head setup. Heads
    # are not diffusion checkpoints and must never be routed to diffusion_models.
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

    # Approximate preview VAEs are intentionally separate from final VAEs in
    # ComfyUI. TAEH3 is the important H3 case, but keep the rule generic.
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

    # The safe target check normalizes the basename and guarantees containment.
    # Keep model installs model-like rather than allowing the endpoint to become
    # a general-purpose file writer.
    if target.suffix.lower() not in service.MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported model file extension: {target.suffix or '<none>'}")

    size = int(asset.get("size_bytes") or 0)
    if size < 0:
        raise ValueError("Asset size cannot be negative.")

    return {
        **asset,
        "provider": provider,
        "destination": destination,
        "filename": target.name,
        "download_url": download_url,
    }


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
    return _ORIGINAL_DOWNLOAD_ASSETS(
        validated,
        hf_token=hf_token,
        civitai_api_key=civitai_api_key,
        force=force,
        progress_callback=progress_callback,
    )


# Patch the function imported by UniversalAssetDownloader so standalone node
# installs and external integrations share exactly the same safety gate.
service.download_assets = secure_download_assets


def _models_root() -> str:
    return str(service.models_dir())


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
    return web.json_response(
        {
            "ok": True,
            "name": "Universal Asset Downloader",
            "version": UAD_VERSION,
            "models_dir": _models_root(),
            "capabilities": {
                "analyze": True,
                "verify": True,
                "install": True,
                "provider_hashes": True,
                "atomic_downloads": True,
                "external_integration": True,
                "nonblocking_analysis": True,
                "rich_progress": True,
                "pdd_heads": True,
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

        results: list[dict[str, Any]] = []
        for file_zero_index, asset in enumerate(validated):
            file_index = file_zero_index + 1

            def progress(downloaded: int, total: int, current_asset: dict[str, Any]) -> None:
                if total_expected > 0:
                    overall = (completed_before + min(downloaded, total)) / total_expected * 100.0
                else:
                    # Unknown aggregate size: still report the per-file fraction,
                    # folded into the batch so the UI visibly advances.
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

            starting_progress = (file_zero_index / max(1, file_count)) * 100.0
            _send_external_progress(
                node_id,
                f"Preparing {asset.get('filename', 'model')}…",
                starting_progress,
                asset,
                downloaded_bytes=0,
                total_bytes=int(asset.get("size_bytes") or 0),
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
            completed_before += int(asset.get("size_bytes") or 0)

        _send_external_progress(
            node_id,
            "Install complete",
            100.0,
            downloaded_bytes=total_expected if total_expected else None,
            total_bytes=total_expected if total_expected else None,
            file_index=file_count if file_count else None,
            file_count=file_count if file_count else None,
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
                }
            )
        return web.json_response({"ok": True, "results": compact})
    except Exception as exc:
        _send_external_progress(node_id, f"Install failed: {exc}", 0.0)
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


__all__ = [
    "UAD_VERSION",
    "secure_download_assets",
    "validate_install_asset",
]
