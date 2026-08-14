from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _Routes:
    def __init__(self):
        self.registered = []

    def get(self, path):
        def decorator(fn):
            self.registered.append(("GET", path, fn))
            return fn
        return decorator

    def post(self, path):
        def decorator(fn):
            self.registered.append(("POST", path, fn))
            return fn
        return decorator


routes = _Routes()
server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(
    instance=types.SimpleNamespace(routes=routes, send_sync=lambda *_args, **_kwargs: None)
)
sys.modules["server"] = server

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = str(Path.cwd() / ".test-models")
sys.modules["folder_paths"] = folder_paths

root = Path(__file__).resolve().parents[1]
nodes_dir = root / "nodes"
nodes_pkg = types.ModuleType("nodes")
nodes_pkg.__path__ = [str(nodes_dir)]
sys.modules["nodes"] = nodes_pkg

service_spec = importlib.util.spec_from_file_location("nodes.smart_asset_service", nodes_dir / "smart_asset_service.py")
service = importlib.util.module_from_spec(service_spec)
sys.modules["nodes.smart_asset_service"] = service
assert service_spec.loader is not None
service_spec.loader.exec_module(service)

integration_spec = importlib.util.spec_from_file_location("nodes.integration_api", nodes_dir / "integration_api.py")
integration = importlib.util.module_from_spec(integration_spec)
sys.modules["nodes.integration_api"] = integration
assert integration_spec.loader is not None
integration_spec.loader.exec_module(integration)


def run():
    assert "vae_approx" in service.ALLOWED_DESTINATIONS
    assert service.infer_destination(
        "Kijai/MiniMax-H3-TAE",
        "taeh3.safetensors",
    )["destination"] == "vae_approx"

    valid = integration.validate_install_asset(
        {
            "provider": "huggingface",
            "download_url": "https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors?download=true",
            "destination": "vae_approx",
            "filename": "taeh3.safetensors",
            "size_bytes": 123,
        }
    )
    assert valid["destination"] == "vae_approx"

    try:
        integration.validate_install_asset(
            {
                "provider": "huggingface",
                "download_url": "https://evil.example/model.safetensors",
                "destination": "loras",
                "filename": "model.safetensors",
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("untrusted host should be rejected")

    assert service.download_assets is integration.secure_download_assets
    registered = {(method, path) for method, path, _fn in routes.registered}
    assert ("GET", "/uad/status") in registered
    assert ("POST", "/uad/install") in registered
    assert ("POST", "/uad/analyze") in registered
    assert ("POST", "/uad/verify") in registered

    print("integration checks passed")


if __name__ == "__main__":
    run()
