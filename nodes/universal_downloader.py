from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from server import PromptServer

from .smart_asset_service import analyze_url, download_assets, human_size


MANUAL_DESTINATIONS = {
    "Checkpoint": ("Checkpoint", "checkpoints"),
    "LoRA": ("LoRA", "loras"),
    "VAE": ("VAE", "vae"),
    "ControlNet": ("ControlNet", "controlnet"),
    "Upscale Model": ("Upscale Model", "upscale_models"),
    "CLIP": ("Text Encoder", "text_encoders"),
    "UNET": ("Diffusion Model", "diffusion_models"),
    "TextualInversion": ("Embedding", "embeddings"),
}

TRUSTED_DOWNLOAD_HOSTS = {
    "huggingface": {"huggingface.co", "www.huggingface.co"},
    "civitai": {"civitai.com", "www.civitai.com"},
}


class UniversalAssetDownloader:
    """Safe, metadata-aware model installer for ComfyUI."""

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("download_message",)
    FUNCTION = "download_asset"
    CATEGORY = "utilities/downloaders"
    DESCRIPTION = (
        "Analyze Hugging Face and Civitai model links before downloading, infer the correct "
        "ComfyUI model folder, show file sizes, and verify downloads using provider hashes when available."
    )

    def __init__(self):
        self.node_id = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset_url": ("STRING", {"multiline": True, "default": ""}),
                "asset_type": (["Auto", *MANUAL_DESTINATIONS.keys()], {"default": "Auto"}),
                "force_download": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "selection_json": ("STRING", {"multiline": False, "default": ""}),
                "civitai_api_key": ("STRING", {"multiline": False, "default": ""}),
                "hf_token": ("STRING", {"multiline": False, "default": ""}),
                # Retained so old workflows still deserialize. UAD v2 always writes beneath ComfyUI/models.
                "base_path": ("STRING", {"default": "./"}),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    def _send_status(self, status: str, *, progress: float | None = None, asset: dict[str, Any] | None = None):
        if not self.node_id:
            return
        payload = {"node": str(self.node_id), "status": status}
        if progress is not None:
            payload["progress"] = max(0.0, min(100.0, float(progress)))
            PromptServer.instance.send_sync(
                "progress",
                {"node": self.node_id, "value": payload["progress"], "max": 100},
            )
        if asset:
            payload["filename"] = asset.get("filename", "")
            payload["destination"] = asset.get("destination", "")
        PromptServer.instance.send_sync("uad-progress", payload)

    def _progress(self, downloaded: int, total: int, asset: dict[str, Any]):
        percent = (downloaded / total * 100.0) if total else 0.0
        self._send_status(
            f"Downloading {asset.get('filename', 'asset')} · {percent:.1f}%",
            progress=percent,
            asset=asset,
        )

    @staticmethod
    def _decode_selection(selection_json: str) -> list[dict[str, Any]]:
        if not str(selection_json or "").strip():
            return []
        payload = json.loads(selection_json)
        if not isinstance(payload, list):
            raise ValueError("Installer selection is invalid. Re-run Analyze in the node UI.")
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _validate_selected_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean = []
        for asset in assets:
            provider = str(asset.get("provider") or "").lower()
            if provider == "mega":
                raise ValueError(
                    "MEGA does not provide enough trusted metadata for the safe v2 installer yet. "
                    "Use Hugging Face/Civitai for verified installs, or install a MEGA file manually."
                )
            allowed_hosts = TRUSTED_DOWNLOAD_HOSTS.get(provider)
            if not allowed_hosts:
                raise ValueError(f"Unsupported provider in installer selection: {provider or 'unknown'}")
            download_url = str(asset.get("download_url") or "")
            host = (urlparse(download_url).hostname or "").lower()
            if host not in allowed_hosts:
                raise ValueError(f"Refusing untrusted download host {host or '<missing>'} for {provider}.")
            filename = Path(str(asset.get("filename") or "")).name
            if not filename or filename in {".", ".."}:
                raise ValueError("Installer selection contains an unsafe filename.")
            clean.append({**asset, "filename": filename})
        return clean

    @staticmethod
    def _manual_override(assets: list[dict[str, Any]], asset_type: str) -> list[dict[str, Any]]:
        if asset_type == "Auto":
            return assets
        role, destination = MANUAL_DESTINATIONS[asset_type]
        return [
            {
                **asset,
                "asset_type": role,
                "destination": destination,
                "confidence": 1.0,
                "reason": f"manual override: {asset_type}",
            }
            for asset in assets
        ]

    def download_asset(
        self,
        asset_url,
        asset_type,
        force_download,
        node_id,
        selection_json="",
        civitai_api_key="",
        hf_token="",
        base_path="./",
    ):
        del base_path  # kept only for old workflow compatibility
        self.node_id = node_id
        try:
            url = str(asset_url or "").strip()
            if not url:
                return ("Paste a Hugging Face, Civitai, or MEGA model URL first.",)

            selected = self._decode_selection(selection_json)
            if not selected:
                analysis = analyze_url(url, hf_token=hf_token, civitai_api_key=civitai_api_key)
                assets = analysis.get("assets") or []
                if len(assets) == 1:
                    selected = assets
                else:
                    primary = [item for item in assets if item.get("primary")]
                    if len(primary) == 1:
                        selected = primary
                    else:
                        return (
                            f"Found {len(assets)} model files ({analysis.get('total_size_label', 'size unknown')}). "
                            "Use Analyze in the v2 node UI and choose exactly what you want to install; "
                            "the downloader will not guess across a multi-file repository.",
                        )

            selected = self._validate_selected_assets(selected)
            selected = self._manual_override(selected, asset_type)
            total_bytes = sum(int(item.get("size_bytes") or 0) for item in selected)
            self._send_status(
                f"Installing {len(selected)} asset{'s' if len(selected) != 1 else ''} · {human_size(total_bytes)}",
                progress=0,
            )
            results = download_assets(
                selected,
                hf_token=hf_token,
                civitai_api_key=civitai_api_key,
                force=bool(force_download),
                progress_callback=self._progress,
            )

            for result in results:
                if result.get("ok"):
                    continue
                # Never leave a newly downloaded file behind if post-download format verification failed.
                if not result.get("skipped") and result.get("path"):
                    Path(result["path"]).unlink(missing_ok=True)
                raise ValueError(result.get("message") or "Downloaded file failed verification.")

            self._send_status("Install complete", progress=100)
            lines = [f"Installed/verified {len(results)} asset{'s' if len(results) != 1 else ''}:"]
            for result in results:
                asset = result.get("asset") or {}
                state = "already verified" if result.get("skipped") else "downloaded + verified"
                lines.append(
                    f"✓ {asset.get('filename')} → models/{asset.get('destination')} "
                    f"({asset.get('size_label', 'size unknown')}, {state})"
                )
            return ("\n".join(lines),)
        except Exception as exc:
            self._send_status(f"Install failed: {exc}", progress=0)
            return (f"Install failed: {exc}",)
