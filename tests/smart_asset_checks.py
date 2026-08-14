from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import types
from pathlib import Path


class _Routes:
    def post(self, _path):
        return lambda fn: fn


server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
sys.modules["server"] = server

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = str(Path.cwd() / ".test-models")
sys.modules["folder_paths"] = folder_paths

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "smart_asset_service.py"
spec = importlib.util.spec_from_file_location("smart_asset_service", MODULE_PATH)
service = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(service)


def run():
    assert service.infer_destination(
        "Kijai/MiniMax-H3_comfy",
        "minimax_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors",
    )["destination"] == "loras"
    assert service.infer_destination(
        "Kijai/MiniMax-H3_comfy",
        "minimax_h3_video_vae_int8_convrot.safetensors",
    )["destination"] == "vae"
    assert service.infer_destination(
        "Kijai/MiniMax-H3_comfy",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    )["destination"] == "text_encoders"
    assert service.infer_destination(
        "Kijai/MiniMax-H3-experimental",
        "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    )["destination"] == "diffusion_models"

    unknown = service.infer_destination("someone/unknown", "mystery.safetensors")
    assert unknown["destination"] == "unclassified"
    assert unknown["confidence"] < 0.5

    assert service._hf_parse("https://huggingface.co/Kijai/MiniMax-H3_comfy") == (
        "Kijai/MiniMax-H3_comfy",
        "main",
        None,
        None,
    )
    assert service._hf_parse(
        "https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras"
    ) == ("Kijai/MiniMax-H3_comfy", "main", None, "loras")
    assert service._hf_parse(
        "https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/model.safetensors?download=true"
    ) == ("Kijai/MiniMax-H3_comfy", "main", "loras/model.safetensors", None)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        service.folder_paths.models_dir = str(root)
        target = service.safe_target("loras", "../../evil.safetensors")
        assert target == (root / "loras" / "evil.safetensors").resolve()
        try:
            service.safe_target("../../outside", "model.safetensors")
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe destination should be rejected")

        tensor = root / "loras" / "tiny.safetensors"
        tensor.parent.mkdir(parents=True)
        header = json.dumps({"weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        tensor.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")
        digest = hashlib.sha256(tensor.read_bytes()).hexdigest()
        result = service.verify_file(tensor, tensor.stat().st_size, digest)
        assert result["ok"] is True
        assert result["status"] == "verified"

    print("smart asset checks passed")


if __name__ == "__main__":
    run()
