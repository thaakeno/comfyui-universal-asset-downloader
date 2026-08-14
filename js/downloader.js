import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TARGET = "UniversalAssetDownloader";
const DESTINATIONS = [
  ["diffusion_models", "Diffusion models"],
  ["checkpoints", "Checkpoints"],
  ["loras", "LoRAs"],
  ["vae", "VAE"],
  ["text_encoders", "Text encoders"],
  ["clip_vision", "CLIP vision"],
  ["controlnet", "ControlNet"],
  ["upscale_models", "Upscale models"],
  ["embeddings", "Embeddings"],
  ["style_models", "Style models"],
  ["audio_encoders", "Audio models"],
  ["unclassified", "Unclassified / review"],
];

function injectStyles() {
  if (document.getElementById("uad-v2-style")) return;
  const style = document.createElement("style");
  style.id = "uad-v2-style";
  style.textContent = `
    .uad-v2 { box-sizing:border-box; width:100%; height:100%; padding:12px; color:var(--fg-color,#ddd); font:12px/1.45 Inter,system-ui,sans-serif; overflow:auto; }
    .uad-v2 * { box-sizing:border-box; }
    .uad-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
    .uad-title { font-size:15px; font-weight:700; letter-spacing:.01em; }
    .uad-subtitle { opacity:.65; font-size:11px; margin-top:2px; }
    .uad-provider { display:flex; align-items:center; gap:6px; padding:5px 8px; border:1px solid rgba(255,255,255,.10); border-radius:999px; background:rgba(255,255,255,.04); }
    .uad-provider img { width:16px; height:16px; border-radius:3px; object-fit:contain; }
    .uad-url-row { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:stretch; }
    .uad-url { width:100%; min-height:58px; resize:vertical; border:1px solid rgba(255,255,255,.14); background:rgba(0,0,0,.22); color:inherit; border-radius:8px; padding:9px 10px; outline:none; }
    .uad-url:focus { border-color:rgba(120,170,255,.7); box-shadow:0 0 0 2px rgba(100,150,255,.12); }
    .uad-btn { border:1px solid rgba(255,255,255,.14); background:rgba(255,255,255,.08); color:inherit; border-radius:8px; padding:8px 11px; cursor:pointer; font-weight:650; }
    .uad-btn:hover { background:rgba(255,255,255,.13); }
    .uad-btn:disabled { opacity:.4; cursor:not-allowed; }
    .uad-btn-primary { background:rgba(55,110,220,.35); border-color:rgba(90,145,255,.5); }
    .uad-btn-primary:hover { background:rgba(55,110,220,.5); }
    .uad-toolbar { display:flex; flex-wrap:wrap; gap:7px; margin-top:9px; align-items:center; }
    .uad-spacer { flex:1; }
    .uad-link { color:#8fbaff; text-decoration:none; overflow-wrap:anywhere; }
    .uad-link:hover { text-decoration:underline; }
    .uad-summary { margin:10px 0 8px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; }
    .uad-stat { border:1px solid rgba(255,255,255,.09); background:rgba(255,255,255,.035); border-radius:8px; padding:8px; }
    .uad-stat-label { opacity:.58; font-size:10px; text-transform:uppercase; letter-spacing:.06em; }
    .uad-stat-value { font-weight:700; margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .uad-assets { display:flex; flex-direction:column; gap:7px; }
    .uad-asset { display:grid; grid-template-columns:auto minmax(0,1fr) auto; gap:9px; border:1px solid rgba(255,255,255,.10); background:rgba(0,0,0,.13); border-radius:9px; padding:9px; align-items:start; }
    .uad-check { margin-top:4px; width:15px; height:15px; }
    .uad-file { min-width:0; }
    .uad-file-name { font-weight:650; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:block; }
    .uad-meta { margin-top:4px; display:flex; gap:5px; flex-wrap:wrap; align-items:center; opacity:.78; }
    .uad-chip { display:inline-flex; align-items:center; gap:4px; border:1px solid rgba(255,255,255,.10); border-radius:999px; padding:2px 6px; font-size:10px; background:rgba(255,255,255,.035); }
    .uad-chip-high { border-color:rgba(80,200,130,.34); background:rgba(60,170,110,.10); }
    .uad-chip-medium { border-color:rgba(245,190,80,.35); background:rgba(200,145,45,.10); }
    .uad-chip-low { border-color:rgba(240,105,105,.38); background:rgba(200,65,65,.10); }
    .uad-destination { max-width:160px; border:1px solid rgba(255,255,255,.12); background:#202020; color:inherit; border-radius:6px; padding:4px 5px; font-size:10px; }
    .uad-notice { margin-top:8px; padding:8px 9px; border-radius:8px; background:rgba(255,190,60,.08); border:1px solid rgba(255,190,60,.18); color:rgba(255,255,255,.78); }
    .uad-status { margin-top:9px; min-height:34px; border-radius:8px; background:rgba(0,0,0,.18); border:1px solid rgba(255,255,255,.08); padding:8px 9px; white-space:pre-wrap; overflow-wrap:anywhere; }
    .uad-progress { height:5px; margin-top:7px; border-radius:99px; overflow:hidden; background:rgba(255,255,255,.08); }
    .uad-progress > div { height:100%; width:0%; background:currentColor; opacity:.72; transition:width .15s ease; }
    .uad-details { margin-top:8px; border:1px solid rgba(255,255,255,.08); border-radius:8px; padding:7px 9px; }
    .uad-details summary { cursor:pointer; font-weight:650; }
    .uad-secret-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:8px; }
    .uad-field label { display:block; opacity:.62; margin-bottom:3px; font-size:10px; }
    .uad-field input { width:100%; border:1px solid rgba(255,255,255,.10); border-radius:6px; background:rgba(0,0,0,.2); color:inherit; padding:6px 7px; }
    .uad-empty { opacity:.55; padding:16px 4px; text-align:center; }
  `;
  document.head.appendChild(style);
}

function findWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function hideNativeWidget(widget) {
  if (!widget || widget.__uadHidden) return;
  widget.__uadHidden = true;
  widget.__uadOriginalType = widget.type;
  widget.type = "uad-hidden";
  widget.computeSize = () => [0, -4];
}

function providerInfo(url) {
  const value = String(url || "").toLowerCase();
  if (value.includes("huggingface.co")) return { label: "Hugging Face", icon: "https://huggingface.co/favicon.ico" };
  if (value.includes("civitai.com")) return { label: "Civitai", icon: "https://civitai.com/favicon.ico" };
  if (value.includes("mega.nz")) return { label: "MEGA", icon: "https://mega.nz/favicon.ico" };
  return { label: "Paste a model link", icon: "" };
}

function bytesLabel(bytes) {
  const size = Number(bytes) || 0;
  if (!size) return "Unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = size;
  let unit = units[0];
  for (const candidate of units) {
    unit = candidate;
    if (value < 1024 || candidate === units.at(-1)) break;
    value /= 1024;
  }
  return `${value.toFixed(unit === "GB" || unit === "TB" ? 2 : 1)} ${unit}`;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({ ok: false, error: `HTTP ${response.status}` }));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function installDownloaderUI(node) {
  if (node.__uadV2Installed || typeof node.addDOMWidget !== "function") return;
  node.__uadV2Installed = true;
  injectStyles();

  const native = {
    url: findWidget(node, "asset_url"),
    type: findWidget(node, "asset_type"),
    force: findWidget(node, "force_download"),
    selection: findWidget(node, "selection_json"),
    civitai: findWidget(node, "civitai_api_key"),
    hf: findWidget(node, "hf_token"),
    base: findWidget(node, "base_path"),
  };
  Object.values(native).forEach(hideNativeWidget);
  if (native.type) native.type.value = "Auto";

  const root = document.createElement("div");
  root.className = "uad-v2";
  root.innerHTML = `
    <div class="uad-head">
      <div><div class="uad-title">Universal Asset Downloader</div><div class="uad-subtitle">Analyze first. Install to the right ComfyUI folder. Verify after.</div></div>
      <div class="uad-provider"><img hidden><span>Paste a model link</span></div>
    </div>
    <div class="uad-url-row">
      <textarea class="uad-url" spellcheck="false" placeholder="Hugging Face, Civitai, or MEGA URL"></textarea>
      <button class="uad-btn uad-btn-primary uad-analyze" type="button">Analyze</button>
    </div>
    <div class="uad-toolbar">
      <button class="uad-btn uad-select-all" type="button" disabled>Select all</button>
      <button class="uad-btn uad-clear" type="button" disabled>Clear</button>
      <span class="uad-spacer"></span>
      <a class="uad-link uad-open-source" href="#" target="_blank" rel="noopener noreferrer" hidden>Open source ↗</a>
    </div>
    <div class="uad-summary" hidden>
      <div class="uad-stat"><div class="uad-stat-label">Files found</div><div class="uad-stat-value uad-count">0</div></div>
      <div class="uad-stat"><div class="uad-stat-label">Repository total</div><div class="uad-stat-value uad-total">0 B</div></div>
      <div class="uad-stat"><div class="uad-stat-label">Selected</div><div class="uad-stat-value uad-selected">0 B</div></div>
    </div>
    <div class="uad-assets"><div class="uad-empty">Paste a model URL and click Analyze. Nothing downloads during analysis.</div></div>
    <div class="uad-notice" hidden></div>
    <details class="uad-details">
      <summary>Access tokens & advanced</summary>
      <div class="uad-secret-grid">
        <div class="uad-field"><label>Hugging Face token (optional)</label><input class="uad-hf-token" type="password" autocomplete="off"></div>
        <div class="uad-field"><label>Civitai API key (optional)</label><input class="uad-civitai-token" type="password" autocomplete="off"></div>
      </div>
      <label style="display:flex;gap:7px;align-items:center;margin-top:8px"><input class="uad-force" type="checkbox"> Force replace an existing file only if you intentionally want to redownload it</label>
    </details>
    <div class="uad-toolbar">
      <button class="uad-btn uad-verify" type="button" disabled>Verify installed</button>
      <span class="uad-spacer"></span>
      <button class="uad-btn uad-btn-primary uad-install" type="button" disabled>Install selected</button>
    </div>
    <div class="uad-status">Ready. Analysis is metadata-only.</div>
    <div class="uad-progress"><div></div></div>
  `;

  const el = {
    provider: root.querySelector(".uad-provider"),
    providerImg: root.querySelector(".uad-provider img"),
    providerText: root.querySelector(".uad-provider span"),
    url: root.querySelector(".uad-url"),
    analyze: root.querySelector(".uad-analyze"),
    selectAll: root.querySelector(".uad-select-all"),
    clear: root.querySelector(".uad-clear"),
    openSource: root.querySelector(".uad-open-source"),
    summary: root.querySelector(".uad-summary"),
    count: root.querySelector(".uad-count"),
    total: root.querySelector(".uad-total"),
    selected: root.querySelector(".uad-selected"),
    assets: root.querySelector(".uad-assets"),
    notice: root.querySelector(".uad-notice"),
    hf: root.querySelector(".uad-hf-token"),
    civitai: root.querySelector(".uad-civitai-token"),
    force: root.querySelector(".uad-force"),
    verify: root.querySelector(".uad-verify"),
    install: root.querySelector(".uad-install"),
    status: root.querySelector(".uad-status"),
    progress: root.querySelector(".uad-progress > div"),
  };

  const state = { analysis: null, assets: [], selected: new Set(), busy: false };
  el.url.value = native.url?.value || "";
  el.hf.value = native.hf?.value || "";
  el.civitai.value = native.civitai?.value || "";
  el.force.checked = Boolean(native.force?.value);

  const updateProvider = () => {
    const info = providerInfo(el.url.value);
    el.providerText.textContent = info.label;
    el.providerImg.hidden = !info.icon;
    if (info.icon) el.providerImg.src = info.icon;
  };

  const selectedAssets = () => state.assets.filter((asset) => state.selected.has(asset.id));

  const syncNative = () => {
    if (native.url) native.url.value = el.url.value.trim();
    if (native.hf) native.hf.value = el.hf.value;
    if (native.civitai) native.civitai.value = el.civitai.value;
    if (native.force) native.force.value = Boolean(el.force.checked);
    if (native.type) native.type.value = "Auto";
    if (native.selection) native.selection.value = JSON.stringify(selectedAssets());
  };

  const updateSelectionSummary = () => {
    const chosen = selectedAssets();
    const bytes = chosen.reduce((sum, asset) => sum + (Number(asset.size_bytes) || 0), 0);
    el.selected.textContent = chosen.length ? `${chosen.length} · ${bytesLabel(bytes)}` : "Nothing selected";
    el.install.disabled = !chosen.length || state.busy;
    el.verify.disabled = !chosen.length || state.busy;
    el.clear.disabled = !state.assets.length || state.busy;
    el.selectAll.disabled = !state.assets.length || state.busy;
    syncNative();
  };

  const destinationSelect = (asset) => {
    const select = document.createElement("select");
    select.className = "uad-destination";
    for (const [value, label] of DESTINATIONS) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = `models/${value}`;
      option.title = label;
      if (value === asset.destination) option.selected = true;
      select.appendChild(option);
    }
    select.addEventListener("change", (event) => {
      event.stopPropagation();
      asset.destination = select.value;
      asset.reason = "manual destination override";
      asset.confidence = 1;
      renderAssets();
    });
    return select;
  };

  const renderAssets = () => {
    el.assets.replaceChildren();
    if (!state.assets.length) {
      const empty = document.createElement("div");
      empty.className = "uad-empty";
      empty.textContent = "No model files found.";
      el.assets.appendChild(empty);
      updateSelectionSummary();
      return;
    }
    for (const asset of state.assets) {
      const row = document.createElement("div");
      row.className = "uad-asset";
      const check = document.createElement("input");
      check.type = "checkbox";
      check.className = "uad-check";
      check.checked = state.selected.has(asset.id);
      check.addEventListener("change", () => {
        if (check.checked) state.selected.add(asset.id); else state.selected.delete(asset.id);
        updateSelectionSummary();
      });

      const file = document.createElement("div");
      file.className = "uad-file";
      const link = document.createElement("a");
      link.className = "uad-file-name uad-link";
      link.href = asset.source_url || el.url.value;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = asset.remote_path || asset.filename;
      link.title = asset.remote_path || asset.filename;
      const meta = document.createElement("div");
      meta.className = "uad-meta";
      const confidence = Number(asset.confidence) || 0;
      const confidenceClass = confidence >= .85 ? "uad-chip-high" : confidence >= .55 ? "uad-chip-medium" : "uad-chip-low";
      const confidenceText = confidence >= .85 ? "high confidence" : confidence >= .55 ? "review" : "manual review";
      meta.innerHTML = `
        <span class="uad-chip">${asset.size_label || bytesLabel(asset.size_bytes)}</span>
        <span class="uad-chip">${asset.asset_type || "Unknown"}</span>
        <span class="uad-chip ${confidenceClass}" title="${String(asset.reason || "").replaceAll('"', '&quot;')}">${confidenceText}</span>
        ${asset.sha256 ? '<span class="uad-chip">SHA256 available</span>' : '<span class="uad-chip">size/basic verify</span>'}
      `;
      file.append(link, meta);
      row.append(check, file, destinationSelect(asset));
      el.assets.appendChild(row);
    }
    updateSelectionSummary();
  };

  const setBusy = (busy, message = "") => {
    state.busy = busy;
    el.analyze.disabled = busy;
    updateSelectionSummary();
    if (message) el.status.textContent = message;
  };

  const analyze = async () => {
    const url = el.url.value.trim();
    if (!url) {
      el.status.textContent = "Paste a model URL first.";
      return;
    }
    syncNative();
    setBusy(true, "Reading provider metadata… no files are being downloaded.");
    el.progress.style.width = "0%";
    try {
      const result = await postJson("/uad/analyze", { url, hf_token: el.hf.value, civitai_api_key: el.civitai.value });
      state.analysis = result;
      state.assets = Array.isArray(result.assets) ? result.assets : [];
      state.selected.clear();
      if (state.assets.length === 1) state.selected.add(state.assets[0].id);
      else for (const asset of state.assets) if (asset.primary) state.selected.add(asset.id);
      el.summary.hidden = false;
      el.count.textContent = String(state.assets.length);
      el.total.textContent = result.total_size_label || bytesLabel(result.total_bytes);
      el.notice.hidden = !result.notice;
      el.notice.textContent = result.notice || "";
      el.openSource.hidden = false;
      el.openSource.href = result.source_url || url;
      el.status.textContent = `Analysis complete. ${state.assets.length} file${state.assets.length === 1 ? "" : "s"} found.`;
      renderAssets();
      node.setSize?.([Math.max(node.size?.[0] || 700, 700), Math.min(900, Math.max(560, 410 + state.assets.length * 72))]);
    } catch (error) {
      state.analysis = null;
      state.assets = [];
      state.selected.clear();
      el.summary.hidden = true;
      el.notice.hidden = true;
      el.openSource.hidden = true;
      el.assets.innerHTML = `<div class="uad-empty">Analysis failed. Check the link or token and try again.</div>`;
      el.status.textContent = `Analysis failed: ${error.message}`;
      updateSelectionSummary();
    } finally {
      setBusy(false);
    }
  };

  el.url.addEventListener("input", () => {
    updateProvider();
    if (native.url) native.url.value = el.url.value;
  });
  el.hf.addEventListener("input", syncNative);
  el.civitai.addEventListener("input", syncNative);
  el.force.addEventListener("change", syncNative);
  el.analyze.addEventListener("click", (event) => { event.stopPropagation(); analyze(); });
  el.selectAll.addEventListener("click", (event) => {
    event.stopPropagation();
    state.assets.forEach((asset) => state.selected.add(asset.id));
    renderAssets();
  });
  el.clear.addEventListener("click", (event) => {
    event.stopPropagation();
    state.selected.clear();
    renderAssets();
  });
  el.install.addEventListener("click", async (event) => {
    event.stopPropagation();
    syncNative();
    if (!selectedAssets().length) return;
    el.status.textContent = `Queued ${selectedAssets().length} selected asset${selectedAssets().length === 1 ? "" : "s"} for safe install.`;
    el.progress.style.width = "0%";
    try {
      await app.queuePrompt(0, 1);
    } catch (error) {
      el.status.textContent = `Could not queue install: ${error.message}`;
    }
  });
  el.verify.addEventListener("click", async (event) => {
    event.stopPropagation();
    const items = selectedAssets();
    if (!items.length) return;
    setBusy(true, `Verifying ${items.length} local file${items.length === 1 ? "" : "s"}…`);
    try {
      const result = await postJson("/uad/verify", { items });
      const good = (result.results || []).filter((item) => item.ok).length;
      const bad = (result.results || []).length - good;
      const details = (result.results || []).map((item) => `${item.ok ? "✓" : "✗"} ${item.path?.split(/[\\/]/).at(-1) || "asset"}: ${item.message}`).join("\n");
      el.status.textContent = `${good} verified${bad ? `, ${bad} need attention` : ""}.\n${details}`;
    } catch (error) {
      el.status.textContent = `Verification failed: ${error.message}`;
    } finally {
      setBusy(false);
    }
  });

  node.__uadV2 = { root, el, state, renderAssets };
  const widget = node.addDOMWidget("uad_v2", "uad_v2", root, { serialize: false, hideOnZoom: false });
  widget.computeSize = (width) => [width, Math.max(500, (node.size?.[1] || 620) - 30)];
  node.setSize?.([Math.max(node.size?.[0] || 700, 700), Math.max(node.size?.[1] || 620, 620)]);
  updateProvider();
  updateSelectionSummary();
}

api.addEventListener("uad-progress", ({ detail }) => {
  const node = app.graph?.getNodeById?.(Number(detail?.node)) || app.graph?.getNodeById?.(detail?.node);
  const ui = node?.__uadV2;
  if (!ui) return;
  if (detail?.status) ui.el.status.textContent = detail.status;
  if (Number.isFinite(Number(detail?.progress))) ui.el.progress.style.width = `${Math.max(0, Math.min(100, Number(detail.progress)))}%`;
});

app.registerExtension({
  name: "Comfy.UniversalAssetDownloader.v2",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET) return;
    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      originalExecuted?.apply(this, arguments);
      const ui = this.__uadV2;
      const text = message?.text?.[0] || message?.download_message?.[0];
      if (ui && text) ui.el.status.textContent = text;
    };
  },
  async nodeCreated(node) {
    if (node.comfyClass === TARGET) installDownloaderUI(node);
  },
});
