"""ComfyUI entry point for Universal Asset Downloader."""

# Import integration layers before the node class so the shared downloader is
# patched with the strict safety/Xet gate and the nonblocking provider endpoints.
from .nodes import integration_api as _integration_api
from .nodes import async_api as _async_api  # noqa: F401
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

print(f"✅ Universal Asset Downloader v{_integration_api.UAD_VERSION} loaded successfully!")
