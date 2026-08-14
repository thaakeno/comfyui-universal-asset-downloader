"""Safe destination extension for H3 Studio multimodal analyzer assets."""

from __future__ import annotations

from typing import Any

from . import integration_api
from . import smart_asset_service as service

DESTINATION = "h3studio_vlm"
UAD_VERSION = "2.1.4"

# Keep the same containment/security policy as every other UAD destination. The
# only change is admitting ComfyUI/models/h3studio_vlm as a deliberate model
# folder instead of forcing MiniCPM GGUFs into text_encoders or unclassified.
service.ALLOWED_DESTINATIONS.add(DESTINATION)

_PREVIOUS_INFER = service.infer_destination


def _infer_destination(
    repo_id: str,
    filename: str,
    declared_type: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    text = f"{repo_id} {filename} {' '.join(tags or [])}".lower().replace("-", "_")
    if "minicpm_v_4.6" in text or "minicpm-v-4.6" in text or "minicpm_v4_6" in text:
        if str(filename or "").lower().endswith(".gguf"):
            return {
                "asset_type": "Multimodal VLM GGUF",
                "destination": DESTINATION,
                "confidence": 0.995,
                "reason": "H3 Studio MiniCPM-V 4.6 GGUF/mmproj signature",
            }
    return _PREVIOUS_INFER(repo_id, filename, declared_type, tags)


service.infer_destination = _infer_destination
# /uad/status reads this module global dynamically, so version reporting stays
# accurate without duplicating the existing status route.
integration_api.UAD_VERSION = UAD_VERSION
