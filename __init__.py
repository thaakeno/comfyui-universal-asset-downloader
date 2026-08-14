"""ComfyUI entry point for Universal Asset Downloader."""

# Import the integration layer first. It extends the v2 analyzer with safe
# external-install routes and patches the shared installer safety gate before
# the node class imports those helpers.
from .nodes import integration_api as _integration_api  # noqa: F401
from .nodes.universal_downloader import UniversalAssetDownloader

NODE_CLASS_MAPPINGS = {
    "UniversalAssetDownloader": UniversalAssetDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalAssetDownloader": "🌐 Universal Asset Downloader",
}

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

print("✅ Universal Asset Downloader v2 loaded successfully!")
