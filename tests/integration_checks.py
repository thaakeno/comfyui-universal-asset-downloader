from __future__ import annotations

import importlib.util
import shutil
import struct
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

vlm_spec = importlib.util.spec_from_file_location("nodes.h3studio_vlm", nodes_dir / "h3studio_vlm.py")
vlm = importlib.util.module_from_spec(vlm_spec)
sys.modules["nodes.h3studio_vlm"] = vlm
assert vlm_spec.loader is not None
vlm_spec.loader.exec_module(vlm)

async_spec = importlib.util.spec_from_file_location("nodes.async_api", nodes_dir / "async_api.py")
async_api = importlib.util.module_from_spec(async_spec)
sys.modules["nodes.async_api"] = async_api
assert async_spec.loader is not None
async_spec.loader.exec_module(async_api)

face_spec = importlib.util.spec_from_file_location("nodes.h3studio_face_assets", nodes_dir / "h3studio_face_assets.py")
face_assets = importlib.util.module_from_spec(face_spec)
sys.modules["nodes.h3studio_face_assets"] = face_assets
assert face_spec.loader is not None
face_spec.loader.exec_module(face_assets)


def _write_tiny_safetensors(path: Path) -> None:
    header = b'{"x":{}}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"payload")


def _check_symlink_fast_verify() -> None:
    models = Path(folder_paths.models_dir)
    external = Path.cwd() / ".test-external-uad"
    shutil.rmtree(models, ignore_errors=True)
    shutil.rmtree(external, ignore_errors=True)
    try:
        loras = models / "loras"
        loras.mkdir(parents=True, exist_ok=True)
        real = external / "model.safetensors"
        _write_tiny_safetensors(real)
        logical = loras / "model.safetensors"
        logical.symlink_to(real.resolve())

        result = async_api._verify_fast_one(
            {
                "destination": "loras",
                "filename": logical.name,
                "size_bytes": real.stat().st_size,
                # Deliberately wrong hash: fast verification must not reread and
                # reject the whole file just because a provider SHA is present.
                "sha256": "0" * 64,
            }
        )
        assert result["ok"] is True
        assert result["status"] == "verified_fast"
        assert result["verification_level"] == "fast"
        assert result["symlink"] is True
    finally:
        shutil.rmtree(models, ignore_errors=True)
        shutil.rmtree(external, ignore_errors=True)


def _check_face_refine_assets() -> None:
    assert "ultralytics/bbox" in service.ALLOWED_DESTINATIONS
    assert "sams" in service.ALLOWED_DESTINATIONS

    yolo_target = service.safe_target("ultralytics/bbox", "face_yolov8m.pt")
    assert yolo_target == Path(folder_paths.models_dir).resolve() / "ultralytics" / "bbox" / "face_yolov8m.pt"

    sam = integration.validate_install_asset(
        {
            "provider": "meta",
            "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "destination": "sams",
            "filename": "sam_vit_b_01ec64.pth",
            "size_bytes": 375_042_383,
        }
    )
    assert sam["provider"] == "meta"
    assert sam["destination"] == "sams"
    assert sam["filename"] == "sam_vit_b_01ec64.pth"

    rejected = [
        {
            "provider": "meta",
            "download_url": "https://evil.example/sam_vit_b_01ec64.pth",
            "destination": "sams",
            "filename": "sam_vit_b_01ec64.pth",
        },
        {
            "provider": "meta",
            "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "destination": "sams",
            "filename": "sam_vit_h_4b8939.pth",
        },
        {
            "provider": "meta",
            "download_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "destination": "loras",
            "filename": "sam_vit_b_01ec64.pth",
        },
    ]
    for asset in rejected:
        try:
            integration.validate_install_asset(asset)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe Meta asset should be rejected: {asset}")


def run():
    assert "vae_approx" in service.ALLOWED_DESTINATIONS
    assert "pdd_heads" in service.ALLOWED_DESTINATIONS
    assert "h3studio_vlm" in service.ALLOWED_DESTINATIONS
    assert service.infer_destination(
        "Kijai/MiniMax-H3-TAE",
        "taeh3.safetensors",
    )["destination"] == "vae_approx"
    assert service.infer_destination(
        "unsloth/Qwen3.5-4B-GGUF",
        "Qwen3.5-4B-UD-Q4_K_XL.gguf",
    )["destination"] == "h3studio_vlm"
    assert service.infer_destination(
        "unsloth/Qwen3.5-4B-GGUF",
        "mmproj-BF16.gguf",
    )["destination"] == "h3studio_vlm"
    assert integration.UAD_VERSION == "2.1.5"

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

    qwen = integration.validate_install_asset(
        {
            "provider": "huggingface",
            "download_url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-BF16.gguf?download=true",
            "destination": "h3studio_vlm",
            "filename": "qwen3.5_4b_mmproj_bf16.gguf",
            "size_bytes": 123,
        }
    )
    assert qwen["destination"] == "h3studio_vlm"
    assert qwen["filename"] == "qwen3.5_4b_mmproj_bf16.gguf"

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
    assert ("POST", "/uad/analyze-fast") in registered
    assert ("POST", "/uad/verify-fast") in registered

    _check_face_refine_assets()
    _check_symlink_fast_verify()
    print("integration checks passed")


if __name__ == "__main__":
    run()
