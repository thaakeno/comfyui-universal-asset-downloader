"""Safe H3 Studio Face Refine asset destinations and provider policy.

The additional destinations are fixed allow-list entries under ``ComfyUI/models``.
For optional SAM we also allow only Meta's official HTTPS checkpoint host; this is
not a generic arbitrary-URL escape hatch.
"""

from __future__ import annotations

from urllib.parse import urlparse

from . import integration_api
from . import smart_asset_service as service

FACE_REFINE_DESTINATIONS = {
    "ultralytics/bbox",
    "sams",
}
_META_HOSTS = {"dl.fbaipublicfiles.com"}

service.ALLOWED_DESTINATIONS.update(FACE_REFINE_DESTINATIONS)

_original_validate_install_asset = integration_api.validate_install_asset


def _validate_install_asset(asset):
    provider = str((asset or {}).get("provider") or "").strip().lower()
    if provider != "meta":
        return _original_validate_install_asset(asset)

    download_url = str(asset.get("download_url") or "").strip()
    parsed = urlparse(download_url)
    host = parsed.netloc.lower().split(":", 1)[0]
    if parsed.scheme != "https" or host not in _META_HOSTS:
        raise ValueError("Meta model installs are restricted to dl.fbaipublicfiles.com over HTTPS.")

    destination = str(asset.get("destination") or "").strip().lower()
    filename = str(asset.get("filename") or "").strip()
    target = service.safe_target(destination, filename)
    if target.suffix.lower() not in service.MODEL_EXTENSIONS:
        raise ValueError(f"Unsupported model file extension: {target.suffix or '<none>'}")

    size = int(asset.get("size_bytes") or 0)
    if size < 0:
        raise ValueError("Asset size cannot be negative.")

    return {
        **asset,
        "provider": "meta",
        "destination": destination,
        "filename": target.name,
        "download_url": download_url,
    }


integration_api.validate_install_asset = _validate_install_asset

# secure_download_assets resolves its validator from integration_api globals at
# call time, so replacing it here protects both /uad/install and internal installs.

__all__ = ["FACE_REFINE_DESTINATIONS"]
