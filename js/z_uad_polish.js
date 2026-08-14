import { app } from "/scripts/app.js";

const TARGET = "UniversalAssetDownloader";

function injectPolishStyles() {
  if (document.getElementById("uad-v2-polish-style")) return;
  const style = document.createElement("style");
  style.id = "uad-v2-polish-style";
  style.textContent = `
    .uad-v2{padding:10px!important;overflow:hidden!important;display:flex!important;flex-direction:column!important;gap:8px!important;background:linear-gradient(180deg,rgba(20,24,25,.98),rgba(15,18,19,.98))!important}
    .uad-head{margin:0!important;align-items:center!important}.uad-title{font-size:14px!important}.uad-subtitle{font-size:10px!important;opacity:.58!important}
    .uad-provider{padding:4px 7px!important;font-size:10px!important}.uad-url-row{grid-template-columns:minmax(0,1fr) auto auto!important;gap:6px!important}.uad-url{min-height:42px!important;max-height:68px!important;resize:vertical!important;padding:7px 8px!important}
    .uad-btn{padding:6px 8px!important;border-radius:6px!important;font-size:10px!important}.uad-cancel{border-color:rgba(248,113,113,.35)!important;background:rgba(248,113,113,.08)!important}
    .uad-toolbar{margin-top:0!important;gap:5px!important}.uad-summary{margin:0!important;gap:5px!important}.uad-stat{padding:6px!important}.uad-stat-label{font-size:8px!important}.uad-stat-value{font-size:10px!important}
    .uad-assets{min-height:68px!important;max-height:245px!important;overflow:auto!important;gap:5px!important;padding-right:2px!important}.uad-asset{padding:7px!important;gap:7px!important;border-radius:7px!important}.uad-file-name{font-size:10px!important}.uad-meta{gap:3px!important;margin-top:3px!important}.uad-chip{font-size:8px!important;padding:1px 5px!important}.uad-destination{max-width:142px!important;font-size:9px!important;padding:3px 4px!important}
    .uad-notice{margin:0!important;padding:6px 7px!important;font-size:9px!important}.uad-details{margin:0!important;padding:5px 7px!important;font-size:9px!important}.uad-secret-grid{margin-top:6px!important}.uad-field input{padding:5px 6px!important}
    .uad-status{margin:0!important;min-height:31px!important;max-height:86px!important;overflow:auto!important;padding:6px 7px!important;font-size:9px!important}.uad-progress{margin:0!important;height:4px!important}.uad-empty{padding:13px 4px!important}
    .uad-busy-note{font-size:9px;opacity:.65;white-space:nowrap}.uad-v2.is-busy .uad-provider{opacity:.72}
  `;
  document.head.appendChild(style);
}

function setBusy(ui, busy, text = "") {
  ui.state.busy = busy;
  ui.root.classList.toggle("is-busy", busy);
  ui.el.analyze.disabled = busy;
  ui.el.install.disabled = busy || !ui.state.selected.size;
  ui.el.verify.disabled = busy || !ui.state.selected.size;
  ui.el.selectAll.disabled = busy || !ui.state.assets.length;
  ui.el.clear.disabled = busy || !ui.state.assets.length;
  const cancel = ui.root.querySelector(".uad-cancel");
  if (cancel) cancel.hidden = !busy;
  if (text) ui.el.status.textContent = text;
}

async function postJson(path, payload, signal) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  const data = await response.json().catch(() => ({ ok: false, error: `HTTP ${response.status}` }));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function patchNode(node, attempt = 0) {
  const ui = node.__uadV2;
  if (!ui) {
    if (attempt < 20) setTimeout(() => patchNode(node, attempt + 1), 50);
    return;
  }
  if (ui.__polished) return;
  ui.__polished = true;
  injectPolishStyles();

  node.setSize?.([640, 540]);
  const widget = (node.widgets || []).find((item) => item?.name === "uad_v2");
  if (widget) widget.computeSize = (width) => [width, 500];

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "uad-btn uad-cancel";
  cancel.textContent = "Cancel";
  cancel.hidden = true;
  ui.el.analyze.insertAdjacentElement("afterend", cancel);

  let analyzeController = null;

  const replacementAnalyze = ui.el.analyze.cloneNode(true);
  ui.el.analyze.replaceWith(replacementAnalyze);
  ui.el.analyze = replacementAnalyze;

  replacementAnalyze.addEventListener("click", async (event) => {
    event.stopPropagation();
    const url = ui.el.url.value.trim();
    if (!url) {
      ui.el.status.textContent = "Paste a model URL first.";
      return;
    }

    analyzeController?.abort();
    analyzeController = new AbortController();
    setBusy(ui, true, "Analyzing provider metadata in the background… ComfyUI stays usable.");
    ui.el.progress.style.width = "12%";

    try {
      const result = await postJson(
        "/uad/analyze-fast",
        { url, hf_token: ui.el.hf.value, civitai_api_key: ui.el.civitai.value },
        analyzeController.signal,
      );
      ui.state.analysis = result;
      ui.state.assets = Array.isArray(result.assets) ? result.assets : [];
      ui.state.selected.clear();
      if (ui.state.assets.length === 1) ui.state.selected.add(ui.state.assets[0].id);
      else for (const asset of ui.state.assets) if (asset.primary) ui.state.selected.add(asset.id);

      ui.el.summary.hidden = false;
      ui.el.count.textContent = String(ui.state.assets.length);
      ui.el.total.textContent = result.total_size_label || "Unknown";
      ui.el.notice.hidden = !result.notice;
      ui.el.notice.textContent = result.notice || "";
      ui.el.openSource.hidden = false;
      ui.el.openSource.href = result.source_url || url;
      ui.renderAssets();
      ui.el.progress.style.width = "100%";
      ui.el.status.textContent = `Ready. ${ui.state.assets.length} model file${ui.state.assets.length === 1 ? "" : "s"} found without blocking ComfyUI.`;
      setTimeout(() => { if (!ui.state.busy) ui.el.progress.style.width = "0%"; }, 500);
    } catch (error) {
      if (error?.name === "AbortError") {
        ui.el.status.textContent = "Analysis canceled.";
      } else {
        ui.state.analysis = null;
        ui.state.assets = [];
        ui.state.selected.clear();
        ui.el.summary.hidden = true;
        ui.el.notice.hidden = true;
        ui.el.openSource.hidden = true;
        ui.el.assets.innerHTML = '<div class="uad-empty">Could not analyze this link. Check the URL or access token and try again.</div>';
        ui.el.status.textContent = `Analysis failed: ${error.message}`;
      }
      ui.el.progress.style.width = "0%";
    } finally {
      analyzeController = null;
      setBusy(ui, false);
    }
  });

  cancel.addEventListener("click", (event) => {
    event.stopPropagation();
    analyzeController?.abort();
  });

  const replacementVerify = ui.el.verify.cloneNode(true);
  ui.el.verify.replaceWith(replacementVerify);
  ui.el.verify = replacementVerify;
  replacementVerify.addEventListener("click", async (event) => {
    event.stopPropagation();
    const items = ui.state.assets.filter((asset) => ui.state.selected.has(asset.id));
    if (!items.length) return;
    setBusy(ui, true, `Verifying ${items.length} local file${items.length === 1 ? "" : "s"} in a worker thread…`);
    try {
      const result = await postJson("/uad/verify-fast", { items });
      const rows = result.results || [];
      const good = rows.filter((item) => item.ok).length;
      const details = rows.map((item) => `${item.ok ? "✓" : "✗"} ${item.path || "asset"}\n  ${item.message}`).join("\n");
      ui.el.status.textContent = `${good}/${rows.length} verified.\n${details}`;
    } catch (error) {
      ui.el.status.textContent = `Verification failed: ${error.message}`;
    } finally {
      setBusy(ui, false);
    }
  });
}

app.registerExtension({
  name: "Comfy.UniversalAssetDownloader.v2.polish",
  async nodeCreated(node) {
    if (node.comfyClass === TARGET) patchNode(node);
  },
});
