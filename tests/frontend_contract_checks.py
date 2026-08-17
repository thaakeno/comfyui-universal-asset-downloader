"""Guard the UAD DOM widget against blocking-analysis and layout regressions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLISH = ROOT / "js" / "z_uad_polish.js"


def run() -> None:
    source = POLISH.read_text(encoding="utf-8")
    compact = source.replace(" ", "")

    assert '"/uad/analyze-fast"' in source, "UI analysis must use the worker-thread-backed fast API."
    assert '"/uad/verify-fast"' in source, "UI verification must use the worker-thread-backed fast API."
    assert "new AbortController()" in source, "Provider metadata analysis must remain cancellable."
    assert "resize:none!important" in compact, "URL textarea must not expose the browser resize handle."
    assert "node.setSize?.([640, 540])" not in source, "Do not restore the node/widget sizing that overflowed the node body."
    assert "Comfy.UniversalAssetDownloader.v4.polish" in source, "The active polish layer must keep a unique extension name."
    assert '"base_path"' in source and "hardHideLegacyWidgets" in source, "Legacy base_path must stay hidden from the custom node surface."
    assert 'background:transparent!important' in compact, "The UAD surface must inherit the ComfyUI node surface instead of painting a second dark panel."
    assert "--uad-control:color-mix" in compact, "Controls must derive their colors from the active ComfyUI theme."
    assert not (ROOT / "js" / "uad_v2_polish.js").exists(), "The duplicate legacy polish layer must stay removed."

    print("frontend contract checks passed")


if __name__ == "__main__":
    run()
