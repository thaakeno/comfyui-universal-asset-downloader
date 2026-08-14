from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nodes.hf_xet_download import _download_kwargs, configure_xet_environment


def old_hub(repo_id, filename, *, revision=None, local_dir=None, token=None, force_download=False):
    return "unused"


def new_hub(repo_id, filename, *, revision=None, local_dir=None, token=None, force_download=False, tqdm_class=None):
    return "unused"


def check_signature_bridge() -> None:
    common = dict(
        repo_id="owner/repo",
        remote_path="model.safetensors",
        revision="main",
        stage_dir=Path("/tmp/uad-stage"),
        hf_token="",
        force=False,
        tqdm_class=object,
    )
    old_kwargs, old_direct = _download_kwargs(old_hub, **common)
    assert old_direct is False
    assert "tqdm_class" not in old_kwargs

    new_kwargs, new_direct = _download_kwargs(new_hub, **common)
    assert new_direct is True
    assert new_kwargs["tqdm_class"] is object


def main() -> None:
    old = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "ComfyUI" / "models"
            models.mkdir(parents=True)

            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            os.environ["UAD_HF_XET_HIGH_PERFORMANCE"] = "0"
            os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
            adaptive = configure_xet_environment(models)
            assert adaptive["backend"] == "huggingface_hub+hf_xet"
            assert adaptive["high_performance"] is False
            assert "HF_XET_HIGH_PERFORMANCE" not in os.environ
            assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
            assert Path(adaptive["xet_cache"]).is_dir()

            os.environ["UAD_HF_XET_HIGH_PERFORMANCE"] = "1"
            forced = configure_xet_environment(models)
            assert forced["high_performance"] is True
            assert forced["reason"] == "UAD override"
            assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"

        check_signature_bridge()
        print("xet policy checks: OK")
    finally:
        os.environ.clear()
        os.environ.update(old)


if __name__ == "__main__":
    main()
