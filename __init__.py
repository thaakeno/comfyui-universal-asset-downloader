from .nodes.universal_downloader import UniversalAssetDownloader
import os

# Node mappings
NODE_CLASS_MAPPINGS = { 
    "UniversalAssetDownloader": UniversalAssetDownloader,
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = { 
    "UniversalAssetDownloader": "🌐 Universal Asset Downloader",
}

# Web directory for JavaScript files
WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]

print("✅ Universal Asset Downloader loaded successfully!")