from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nodes.hf_xet_download import configure_xet_environment


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

        print("xet policy checks: OK")
    finally:
        os.environ.clear()
        os.environ.update(old)


if __name__ == "__main__":
    main()
