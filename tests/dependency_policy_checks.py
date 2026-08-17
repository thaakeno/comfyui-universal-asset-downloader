"""Guard dependency bounds that must remain compatible with ComfyUI's Transformers stack."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
HUB_REQUIREMENT = "huggingface_hub>=0.36.2,<1.0"


def run() -> None:
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert HUB_REQUIREMENT in requirements, (
        "requirements.txt must keep huggingface_hub below 1.0 so installing UAD cannot "
        "break Transformers 4.x/ComfyUI environments."
    )

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = set(project.get("dependencies", ()))
    assert HUB_REQUIREMENT in dependencies, (
        "pyproject.toml must publish the same huggingface_hub <1.0 compatibility bound."
    )

    print("dependency policy checks passed")


if __name__ == "__main__":
    run()
