"""Safe destination extension for H3 Studio multimodal analyzer assets."""

from __future__ import annotations

from typing import Any

from . import integration_api
from . import smart_asset_service as service

DESTINATION = "h3studio_vlm"
UAD_VERSION = "2.1.5"

# Keep the same containment/security policy as every other UAD destination. The
# only change is admitting ComfyUI/models/h3studio_vlm as a deliberate model
# folder instead of forcing multimodal GGUFs into diffusion_models/unclassified.
service.ALLOWED_DESTINATIONS.add(DESTINATION)

_PREVIOUS_INFER = service.infer_destination


def _infer_destination(
    repo_id: str,
    filename: str,
    declared_type: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    raw = f"{repo_id} {filename} {' '.join(tags or [])}".lower()
    text = raw.replace("-", "_")
    is_gguf = str(filename or "").lower().endswith(".gguf")

    if is_gguf and ("minicpm_v_4.6" in text or "minicpm-v-4.6" in raw or "minicpm_v4_6" in text):
        return {
            "asset_type": "Multimodal VLM GGUF",
            "destination": DESTINATION,
            "confidence": 0.995,
            "reason": "H3 Studio MiniCPM-V 4.6 GGUF/mmproj signature",
        }

    qwen35 = any(token in raw for token in ("qwen3.5", "qwen-3.5", "qwen3_5")) or "qwen35" in text
    if is_gguf and qwen35:
        return {
            "asset_type": "Multimodal VLM GGUF",
            "destination": DESTINATION,
            "confidence": 0.995,
            "reason": "H3 Studio Qwen3.5 multimodal GGUF/mmproj signature",
        }

    # mmproj-BF16.gguf is generic by filename, but the Qwen3.5 repository ID is
    # present in provider metadata and keeps it unambiguous here.
    if is_gguf and "mmproj" in str(filename or "").lower() and qwen35:
        return {
            "asset_type": "Multimodal VLM Projector",
            "destination": DESTINATION,
            "confidence": 0.999,
            "reason": "Qwen3.5 multimodal projector in a Qwen3.5 GGUF repository",
        }

    return _PREVIOUS_INFER(repo_id, filename, declared_type, tags)


service.infer_destination = _infer_destination
# /uad/status reads this module global dynamically, so version reporting stays
# accurate without duplicating the existing status route.
integration_api.UAD_VERSION = UAD_VERSION
