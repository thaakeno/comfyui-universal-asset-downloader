from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types
from pathlib import Path


class _Routes:
    def post(self, _path):
        return lambda fn: fn


server = types.ModuleType("server")
server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes()))
sys.modules.setdefault("server", server)

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = str(Path.cwd() / ".test-models")
sys.modules.setdefault("folder_paths", folder_paths)

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "smart_asset_service.py"
spec = importlib.util.spec_from_file_location("smart_asset_service", MODULE_PATH)
service = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(service)


def test_h3_routing_is_role_aware():
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


def test_unknown_safetensors_is_quarantined_instead_of_guessed_checkpoint():
    result = service.infer_destination("someone/unknown", "mystery.safetensors")
    assert result["destination"] == "unclassified"
    assert result["confidence"] < 0.5


def test_huggingface_url_parser_supports_repo_tree_and_direct_file():
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


def test_safe_target_never_escapes_models_dir(tmp_path):
    service.folder_paths.models_dir = str(tmp_path)
    target = service.safe_target("loras", "../../evil.safetensors")
    assert target == (tmp_path / "loras" / "evil.safetensors").resolve()
    try:
        service.safe_target("../../outside", "model.safetensors")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe destination should be rejected")


def test_verify_safetensors_header_and_sha256(tmp_path):
    service.folder_paths.models_dir = str(tmp_path)
    target = tmp_path / "loras" / "tiny.safetensors"
    target.parent.mkdir(parents=True)
    header = json.dumps({"weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    target.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00\x00")

    import hashlib

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    result = service.verify_file(target, target.stat().st_size, digest)
    assert result["ok"] is True
    assert result["status"] == "verified"
