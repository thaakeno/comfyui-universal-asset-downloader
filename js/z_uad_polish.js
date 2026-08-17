import { app } from "/scripts/app.js";

const TARGET = "UniversalAssetDownloader";
const ICONS = {
  analyze: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5"></circle><path d="m16 16 4 4M11 8v6M8 11h6"></path></svg>',
  cancel: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17"></path></svg>',
  all: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 12 3 3 5-6M12 12l3 3 5-6"></path></svg>',
  clear: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M9 7V5h6v2M8 10v8M12 10v8M16 10v8M7 7l1 13h8l1-13"></path></svg>',
  external: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 5h5v5M19 5l-8 8"></path><path d="M17 13v5H6V7h5"></path></svg>',
  settings: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"></path><circle cx="16" cy="7" r="2"></circle><circle cx="8" cy="17" r="2"></circle></svg>',
  verify: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 6 6v5c0 4 2.6 7.4 6 9 3.4-1.6 6-5 6-9V6l-6-3Z"></path><path d="m9 12 2 2 4-4"></path></svg>',
  install: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v11m-4-4 4 4 4-4M5 19h14"></path></svg>',
  file: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v14H6Z"></path><path d="M14 3v5h4"></path></svg>',
};

function injectPolishStyles() {
  if (document.getElementById("uad-v3-polish-style")) return;
  const style = document.createElement("style");
  style.id = "uad-v3-polish-style";
  style.textContent = `
    .uad-v2,.uad-v2 *{box-sizing:border-box}
    .uad-v2{width:100%!important;height:100%!important;min-width:0!important;min-height:0!important;padding:10px!important;overflow:hidden!important;display:flex!important;flex-direction:column!important;gap:8px!important;background:linear-gradient(180deg,rgba(24,26,29,.98),rgba(18,20,23,.98))!important;border:1px solid rgba(255,255,255,.055)!important;border-radius:9px!important;font:11px/1.4 Inter,ui-sans-serif,system-ui,sans-serif!important}
    .uad-head{margin:0!important;align-items:flex-start!important;gap:12px!important;min-width:0!important}.uad-head>div:first-child{min-width:0!important}.uad-title{font-size:14px!important;line-height:1.2!important;font-weight:720!important}.uad-subtitle{font-size:10px!important;opacity:.58!important;margin-top:3px!important;max-width:460px!important}
    .uad-provider{flex:0 0 auto!important;max-width:170px!important;min-height:27px!important;padding:4px 8px!important;border-radius:7px!important;font-size:10px!important;background:rgba(255,255,255,.035)!important}.uad-provider span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.uad-provider img{width:15px!important;height:15px!important}
    .uad-url-row{grid-template-columns:minmax(0,1fr) auto auto!important;gap:6px!important;min-width:0!important;align-items:stretch!important}.uad-url{width:100%!important;height:38px!important;min-height:38px!important;max-height:38px!important;resize:none!important;overflow:auto!important;padding:9px 10px!important;border-radius:7px!important;background:rgba(0,0,0,.22)!important;font:10.5px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace!important}.uad-url:focus{border-color:rgba(127,168,255,.62)!important;box-shadow:0 0 0 2px rgba(98,143,240,.10)!important}
    .uad-btn,.uad-open-source{min-height:30px!important;display:inline-flex!important;align-items:center!important;justify-content:center!important;gap:6px!important;padding:6px 9px!important;border-radius:7px!important;font:650 10px/1 Inter,ui-sans-serif,system-ui,sans-serif!important;white-space:nowrap!important;text-decoration:none!important}.uad-btn svg,.uad-open-source svg,.uad-details summary svg,.uad-target-path svg{width:14px;height:14px;flex:none;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}.uad-btn-primary{background:rgba(67,111,214,.28)!important;border-color:rgba(110,151,245,.34)!important}.uad-btn-primary:hover:not(:disabled){background:rgba(67,111,214,.40)!important}.uad-cancel{border-color:rgba(235,112,112,.25)!important;background:rgba(220,80,80,.08)!important}.uad-cancel[hidden],.uad-open-source[hidden]{display:none!important}
    .uad-toolbar{margin:0!important;gap:5px!important;min-width:0!important}.uad-summary{margin:0!important;gap:5px!important}.uad-stat{padding:6px 8px!important;border-radius:7px!important}.uad-stat-label{font-size:8px!important;letter-spacing:.075em!important}.uad-stat-value{font-size:10px!important}
    .uad-assets{flex:1 1 auto!important;min-height:74px!important;max-height:none!important;min-width:0!important;overflow:auto!important;gap:5px!important;padding:1px 2px 1px 0!important;scrollbar-gutter:stable}.uad-assets::-webkit-scrollbar,.uad-status::-webkit-scrollbar{width:7px;height:7px}.uad-assets::-webkit-scrollbar-thumb,.uad-status::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:99px}
    .uad-asset{grid-template-columns:auto minmax(0,1fr) minmax(116px,148px)!important;gap:7px!important;padding:7px!important;border-radius:7px!important;min-width:0!important;background:rgba(0,0,0,.13)!important}.uad-file{min-width:0!important}.uad-file-name{font-size:10.5px!important}.uad-meta{gap:3px!important;margin-top:4px!important}.uad-chip{font-size:8px!important;padding:1px 5px!important;border-radius:5px!important}.uad-destination{width:100%!important;min-width:0!important;max-width:none!important;font-size:9px!important;padding:4px 5px!important;border-radius:6px!important;background:#202226!important}.uad-target-path{display:flex;align-items:center;gap:4px;min-width:0;margin-top:5px;opacity:.53;font:9px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace}.uad-target-path span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .uad-notice{margin:0!important;padding:6px 8px!important;border-radius:7px!important;font-size:9px!important}.uad-details{margin:0!important;padding:0!important;border-radius:7px!important;overflow:hidden!important}.uad-details summary{display:flex!important;align-items:center!important;gap:6px!important;padding:6px 8px!important;font-size:9.5px!important;list-style:none!important}.uad-details summary::-webkit-details-marker{display:none}.uad-secret-grid{margin:6px 8px 0!important;gap:7px!important}.uad-field input{padding:5px 6px!important;font-size:9px!important}.uad-details>label{margin:7px 8px 8px!important;font-size:8.5px!important;opacity:.72!important}
    .uad-status{margin:0!important;min-height:30px!important;max-height:58px!important;overflow:auto!important;padding:6px 8px!important;border-radius:7px!important;font-size:9px!important}.uad-progress{margin:0!important;height:3px!important}.uad-empty{min-height:70px!important;padding:12px!important;font-size:9.5px!important}
    .uad-v2.is-busy .uad-provider{opacity:.55!important}
  `;
  document.head.appendChild(style);
}

function iconize(element, icon, label) {
  if (!element) return;
  element.innerHTML = `${icon}<span>${label}</span>`;
}

function targetLabel(asset) {
  return `models/${asset.destination || "unclassified"}/${asset.filename || "model"}`;
}

function selectedAssets(ui) {
  return ui.state.assets.filter((asset) => ui.state.selected.has(asset.id));
}

function setBusy(ui, busy, text = "") {
  ui.state.busy = busy;
  ui.root.classList.toggle("is-busy", busy);
  ui.el.analyze.disabled = busy;
  ui.el.install.disabled = busy || !ui.state.selected.size;
  ui.el.verify.disabled = busy || !ui.state.selected.size;
  ui.el.selectAll.disabled = busy || !ui.state.assets.length;
  ui.el.clear.disabled = busy || !ui.state.assets.length;
  if (ui.el.cancel) ui.el.cancel.hidden = !busy;
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

function patchRows(ui) {
  const rows = [...ui.el.assets.querySelectorAll(".uad-asset")];
  rows.forEach((row, index) => {
    const asset = ui.state.assets[index];
    if (!asset) return;

    const select = row.querySelector("select.uad-destination");
    if (select && ![...select.options].some((option) => option.value === "vae_approx")) {
      const option = document.createElement("option");
      option.value = "vae_approx";
      option.textContent = "models/vae_approx";
      option.title = "Preview / approximate VAE";
      select.appendChild(option);
    }
    if (select && asset.destination === "vae_approx") select.value = "vae_approx";

    const file = row.querySelector(".uad-file");
    if (!file) return;
    let target = file.querySelector(".uad-target-path");
    if (!target) {
      target = document.createElement("div");
      target.className = "uad-target-path";
      target.innerHTML = `${ICONS.file}<span></span>`;
      file.appendChild(target);
    }
    const label = targetLabel(asset);
    target.querySelector("span").textContent = label;
    target.title = `Expected ComfyUI path: ${label}`;
  });
}

function patchNode(node, attempt = 0) {
  const ui = node.__uadV2;
  if (!ui) {
    if (attempt < 40) setTimeout(() => patchNode(node, attempt + 1), 50);
    return;
  }
  if (ui.__uadV3Polished) return;
  ui.__uadV3Polished = true;
  injectPolishStyles();

  // The previous 540px node / 500px DOM widget left too little room for the
  // LiteGraph title and output row, so the widget visibly overflowed the node.
  node.setSize?.([Math.max(node.size?.[0] || 680, 680), Math.max(node.size?.[1] || 650, 650)]);
  const widget = (node.widgets || []).find((item) => item?.name === "uad_v2");
  if (widget) widget.computeSize = (width) => [Math.max(0, width), Math.max(430, (node.size?.[1] || 650) - 92)];

  iconize(ui.el.analyze, ICONS.analyze, "Analyze");
  iconize(ui.el.selectAll, ICONS.all, "Select all");
  iconize(ui.el.clear, ICONS.clear, "Clear");
  iconize(ui.el.verify, ICONS.verify, "Verify");
  iconize(ui.el.install, ICONS.install, "Install selected");
  if (ui.el.openSource) iconize(ui.el.openSource, ICONS.external, "Open source");
  const summary = ui.root.querySelector(".uad-details summary");
  if (summary) summary.innerHTML = `${ICONS.settings}<span>Access tokens & advanced</span>`;

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "uad-btn uad-cancel";
  cancel.hidden = true;
  iconize(cancel, ICONS.cancel, "Cancel");
  ui.el.analyze.insertAdjacentElement("afterend", cancel);
  ui.el.cancel = cancel;

  let analyzeController = null;

  // Detach the base click handler that calls /uad/analyze. The fast endpoint
  // runs provider work via asyncio.to_thread so ComfyUI's aiohttp/event loop
  // remains responsive while remote metadata is being resolved.
  const replacementAnalyze = ui.el.analyze.cloneNode(true);
  ui.el.analyze.replaceWith(replacementAnalyze);
  ui.el.analyze = replacementAnalyze;

  replacementAnalyze.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    const url = ui.el.url.value.trim();
    if (!url) {
      ui.el.status.textContent = "Paste a model URL first.";
      return;
    }

    analyzeController?.abort();
    const controller = new AbortController();
    analyzeController = controller;
    setBusy(ui, true, "Analyzing provider metadata in a worker thread… ComfyUI stays usable.");
    ui.el.progress.style.width = "12%";

    try {
      const result = await postJson(
        "/uad/analyze-fast",
        { url, hf_token: ui.el.hf.value, civitai_api_key: ui.el.civitai.value },
        controller.signal,
      );
      if (controller.signal.aborted) return;

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
      patchRows(ui);
      ui.el.progress.style.width = "100%";
      ui.el.status.textContent = `Ready. ${ui.state.assets.length} model file${ui.state.assets.length === 1 ? "" : "s"} found.`;
      setTimeout(() => { if (!ui.state.busy) ui.el.progress.style.width = "0%"; }, 550);
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
      if (analyzeController === controller) analyzeController = null;
      setBusy(ui, false);
    }
  });

  cancel.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    analyzeController?.abort();
  });

  // Replace the original verifier with the worker-thread-backed fast verifier.
  const replacementVerify = ui.el.verify.cloneNode(true);
  ui.el.verify.replaceWith(replacementVerify);
  ui.el.verify = replacementVerify;
  replacementVerify.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    const items = selectedAssets(ui);
    if (!items.length || ui.state.busy) return;
    setBusy(ui, true, `Verifying ${items.length} local file${items.length === 1 ? "" : "s"} in a worker thread…`);
    try {
      const result = await postJson("/uad/verify-fast", { items });
      const rows = result.results || [];
      const good = rows.filter((item) => item.ok).length;
      const details = rows.map((item, index) => {
        const path = item.path || targetLabel(items[index] || {});
        return `${item.ok ? "OK" : "CHECK"} · ${path}\n${item.message || ""}`;
      }).join("\n\n");
      ui.el.status.textContent = `${good}/${rows.length} verified.${details ? `\n${details}` : ""}`;
    } catch (error) {
      ui.el.status.textContent = `Verification failed: ${error.message}`;
    } finally {
      setBusy(ui, false);
    }
  });

  patchRows(ui);
  const observer = new MutationObserver(() => patchRows(ui));
  observer.observe(ui.el.assets, { childList: true });
  node.__uadV3PolishObserver = observer;
}

app.registerExtension({
  name: "Comfy.UniversalAssetDownloader.v3.polish",
  async nodeCreated(node) {
    if (node.comfyClass === TARGET || node.type === TARGET) patchNode(node);
  },
});
