"""Guard the authoritative UAD frontend against blocking-analysis and layout regressions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "js" / "downloader.js"


def run() -> None:
    source = FRONTEND.read_text(encoding="utf-8")
    compact = source.replace(" ", "")

    assert '"/uad/analyze-fast"' in source, "UI analysis must use the worker-thread-backed fast API."
    assert '"/uad/verify-fast"' in source, "UI verification must use the worker-thread-backed fast API."
    assert '"/uad/install"' in source, "UI install must use the direct safe install API."
    assert "new AbortController()" in source, "Provider metadata analysis must remain cancellable."
    assert "ANALYZE_CONCURRENCY = 3" in source, "Batch analysis must stay bounded instead of flooding providers."
    assert "MAX_SOURCES = 24" in source, "Batch source count must stay bounded."
    assert "extractUrls" in source and "mapLimit" in source, "Multi-link analysis must remain explicit and bounded."
    assert "resize:none!important" in compact, "URL textarea must not expose the browser resize handle."
    assert '"base_path"' in source and "hardHideLegacyWidgets" in source, "Legacy base_path must stay hidden from the custom node surface."
    assert "Comfy.UniversalAssetDownloader.UI" in source, "The authoritative UI extension name must remain stable."
    assert "uad-authoritative-ui-v5" in source, "The authoritative UI stylesheet must be installed directly by downloader.js."
    assert "🌐 Universal Asset Downloader" in source, "Keep the existing globe branding in the UAD surface."
    assert "uad-sources" in source and "uad-source-count" in source, "Batch sources must remain visible in the UI."
    assert not (ROOT / "js" / "z_uad_polish.js").exists(), "Do not reintroduce a late UI patch layer."
    assert not (ROOT / "js" / "uad_v2_polish.js").exists(), "The duplicate legacy polish layer must stay removed."

    print("frontend contract checks passed")


if __name__ == "__main__":
    run()
