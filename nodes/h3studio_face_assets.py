"""Safe nested destinations used by H3 Studio Face Refine.

These are fixed allow-list entries, not arbitrary user paths. ``safe_target`` and
``_safe_verify_target`` still resolve them under ComfyUI/models and reject any
escape from that root.
"""

from __future__ import annotations

from . import smart_asset_service as service

FACE_REFINE_DESTINATIONS = {
    "ultralytics/bbox",
    "sams",
}

service.ALLOWED_DESTINATIONS.update(FACE_REFINE_DESTINATIONS)

__all__ = ["FACE_REFINE_DESTINATIONS"]
