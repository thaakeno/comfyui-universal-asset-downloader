import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TARGET = "UniversalAssetDownloader";
const MAX_SOURCES = 24;
const ANALYZE_CONCURRENCY = 3;
const MAX_INSTALL_ITEMS = 64;

const DESTINATIONS = [
  ["diffusion_models", "Diffusion models"],
  ["checkpoints", "Checkpoints"],
  ["loras", "LoRAs"],
  ["vae", "VAE"],
  ["vae_approx", "Preview VAE"],
  ["text_encoders", "Text encoders"],
  ["clip_vision", "CLIP vision"],
  ["controlnet", "ControlNet"],
  ["upscale_models", "Upscale models"],
  ["embeddings", "Embeddings"],
  ["style_models", "Style models"],
  ["audio_encoders", "Audio models"],
  ["pdd_heads", "PDD heads"],
  ["unclassified", "Unclassified / review"],
];

const LEGACY_WIDGETS = new Set([
  "asset_url",
  "asset_type",
  "force_download",
  "selection_json",
  "civitai_api_key",
  "hf_token",
  "base_path",
]);

const ICONS = {
  analyze: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"></circle><path d="m15.5 15.5 4.5 4.5M10.5 7.5v6M7.5 10.5h6"></path></svg>',
  cancel: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17"></path></svg>',
  selectAll: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 12 3 3 5-6M12 12l3 3 5-6"></path></svg>',
  clear: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M9 7V5h6v2M8 10v8M12 10v8M16 10v8M7 7l1 13h8l1-13"></path></svg>',
  external: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 5h5v5M19 5l-8 8"></path><path d="M17 13v5H6V7h5"></path></svg>',
  settings: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"></path><circle cx="16" cy="7" r="2"></circle><circle cx="8" cy="17" r="2"></circle></svg>',
  verify: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 6 6v5c0 4 2.6 7.4 6 9 3.4-1.6 6-5 6-9V6l-6-3Z"></path><path d="m9 12 2 2 4-4"></path></svg>',
  install: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v11m-4-4 4 4 4-4M5 19h14"></path></svg>',
  file: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v14H6Z"></path><path d="M14 3v5h4"></path></svg>',
  link: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9.5 14.5 14.5 9.5M7 17H5a4 4 0 0 1 0-8h4M17 7h2a4 4 0 0 1 0 8h-4"></path></svg>',
};

function installStyles() {
  if (document.getElementById("uad-authoritative-ui-v5")) return;
  const style = document.createElement("style");
  style.id = "uad-authoritative-ui-v5";
  style.textContent = `
    .uad-ui,.uad-ui *{box-sizing:border-box;min-width:0}
    .uad-ui{
      --uad-bg:color-mix(in srgb,var(--comfy-input-bg,#15191d) 70%,#080a0c 30%);
      --uad-surface:color-mix(in srgb,var(--comfy-input-bg,#15191d) 90%,#0b0e11 10%);
      --uad-surface-hover:color-mix(in srgb,var(--comfy-input-bg,#15191d) 82%,white 8%);
      --uad-raised:color-mix(in srgb,var(--comfy-input-bg,#15191d) 78%,#536778 22%);
      --uad-line:color-mix(in srgb,var(--border-color,#41484f) 70%,transparent);
      --uad-line-soft:color-mix(in srgb,var(--border-color,#41484f) 38%,transparent);
      --uad-text:var(--input-text,var(--fg-color,#e8ebed));
      --uad-muted:color-mix(in srgb,var(--descrip-text,#7f8992) 94%,white 6%);
      width:100%!important;height:100%!important;min-width:0!important;min-height:0!important;
      display:flex!important;flex-direction:column!important;gap:8px!important;overflow:hidden!important;
      padding:11px 12px 12px!important;border:0!important;border-radius:8px!important;
      background:var(--uad-bg)!important;background-image:none!important;box-shadow:none!important;
      color:var(--uad-text)!important;font:10px/1.42 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif!important;
    }
    .uad-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:0 1px 8px;border-bottom:1px solid var(--uad-line-soft)}
    .uad-title{font-size:13px;font-weight:730;line-height:1.2;color:#f0f2f3}.uad-subtitle{margin-top:3px;max-width:490px;color:var(--uad-muted);font-size:8.6px;line-height:1.35}
    .uad-provider{flex:none;max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:5px 8px;border:1px solid var(--uad-line);border-radius:7px;background:var(--uad-surface);color:#adb6bd;font-size:8.4px}
    .uad-input-row{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:7px;align-items:stretch}.uad-url{width:100%!important;height:72px!important;min-height:72px!important;max-height:72px!important;resize:none!important;overflow:auto!important;padding:8px 10px!important;border:1px solid var(--uad-line)!important;border-radius:8px!important;background:var(--uad-surface)!important;color:var(--uad-text)!important;outline:none!important;box-shadow:none!important;font:9.4px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace!important;white-space:pre!important}.uad-url:hover,.uad-url:focus{background:var(--uad-surface-hover)!important;border-color:#59656f!important}.uad-url:focus{box-shadow:0 0 0 1px rgba(115,133,148,.16)!important}
    .uad-btn,.uad-source-link{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:30px;padding:6px 9px;border:1px solid var(--uad-line);border-radius:7px;background:var(--uad-surface);color:#cbd1d6;text-decoration:none;cursor:pointer;box-shadow:none;font:650 9px/1 Inter,ui-sans-serif,system-ui,sans-serif;white-space:nowrap}.uad-btn:hover:not(:disabled),.uad-source-link:hover{background:var(--uad-surface-hover);border-color:#59656f;color:#f2f4f5}.uad-btn:disabled{opacity:.35;cursor:default}.uad-btn-primary{background:var(--uad-raised);border-color:#566672;color:#f0f2f3}.uad-btn-primary:hover:not(:disabled){background:color-mix(in srgb,var(--uad-raised) 78%,white 8%);border-color:#687681}.uad-cancel{border-color:color-mix(in srgb,#9f6464 42%,var(--uad-line));background:color-mix(in srgb,var(--uad-surface) 93%,#7d4141 7%)}.uad-btn svg,.uad-source-link svg,.uad-details summary svg,.uad-target svg{width:13px;height:13px;flex:none;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
    .uad-input-help{display:flex;align-items:center;gap:6px;min-height:17px;color:var(--uad-muted);font-size:7.8px}.uad-input-help strong{color:#aeb7be;font-weight:650}.uad-toolbar{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.uad-spacer{flex:1}
    .uad-summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--uad-line-soft);border-bottom:1px solid var(--uad-line-soft)}.uad-stat{padding:7px 9px;border-right:1px solid var(--uad-line-soft);background:transparent}.uad-stat:last-child{border-right:0}.uad-stat-label{color:var(--uad-muted);font-size:7.1px;text-transform:uppercase;letter-spacing:.075em}.uad-stat-value{margin-top:2px;color:#dfe3e6;font-size:9.3px;font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .uad-sources{display:flex;gap:5px;overflow:auto;padding-bottom:1px}.uad-source-link{flex:none;max-width:230px;min-height:25px;padding:4px 7px;color:#98a2a9;font-size:7.8px}.uad-source-link span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.uad-source-link.is-error{cursor:default;border-color:rgba(174,91,91,.28);color:#c39b9b;background:transparent}
    .uad-assets{flex:1 1 auto;min-height:98px;display:flex;flex-direction:column;gap:6px;overflow:auto;padding:1px 2px 1px 0;scrollbar-gutter:stable}.uad-asset{display:grid;grid-template-columns:auto minmax(0,1fr) minmax(122px,154px);gap:8px;align-items:start;padding:8px 9px;border:1px solid var(--uad-line);border-radius:8px;background:color-mix(in srgb,var(--uad-surface) 38%,transparent)}.uad-asset:hover{background:color-mix(in srgb,var(--uad-surface-hover) 58%,transparent)}.uad-check{width:14px;height:14px;margin:3px 0 0;accent-color:#8d9ba6}.uad-file{min-width:0}.uad-file-name{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#e4e8ea;text-decoration:none;font-size:9.7px;font-weight:650}.uad-file-name:hover{text-decoration:underline}.uad-meta{display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin-top:4px}.uad-chip{display:inline-flex;align-items:center;min-height:16px;padding:1px 5px;border:1px solid var(--uad-line-soft);border-radius:5px;background:transparent;color:#929ca3;font-size:7.2px}.uad-chip-high{border-color:rgba(87,160,112,.34);color:#a8c9b4}.uad-chip-medium{border-color:rgba(177,139,71,.34);color:#cab88c}.uad-chip-low{border-color:rgba(174,91,91,.34);color:#c89b9b}.uad-target{display:flex;align-items:center;gap:4px;margin-top:5px;color:#77828a;font:8px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace}.uad-target span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.uad-destination{width:100%;padding:5px 6px;border:1px solid var(--uad-line);border-radius:6px;background:var(--uad-surface);color:#d0d5d9;font-size:8.2px;outline:none}.uad-destination:hover,.uad-destination:focus{background:var(--uad-surface-hover);border-color:#59656f}
    .uad-notice,.uad-status{padding:6px 8px;border:1px solid var(--uad-line-soft);border-radius:7px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:8.2px}.uad-notice{max-height:62px;border-color:rgba(172,137,74,.24);background:rgba(172,137,74,.035);color:#b4aa97}.uad-status{min-height:30px;max-height:62px;background:rgba(0,0,0,.08);color:#adb6bd}.uad-details{border:1px solid var(--uad-line);border-radius:7px;overflow:hidden;background:transparent}.uad-details summary{display:flex;align-items:center;gap:6px;padding:6px 8px;cursor:pointer;list-style:none;color:#aeb7be;font-size:8.6px}.uad-details summary:hover{background:rgba(255,255,255,.025)}.uad-details summary::-webkit-details-marker{display:none}.uad-secret-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;padding:2px 8px 7px}.uad-field label{display:block;margin-bottom:3px;color:var(--uad-muted);font-size:7.6px}.uad-field input{width:100%;padding:5px 6px;border:1px solid var(--uad-line);border-radius:6px;background:var(--uad-surface);color:var(--uad-text);outline:none}.uad-field input:focus{border-color:#59656f}.uad-force{display:flex;align-items:center;gap:6px;padding:0 8px 8px;color:var(--uad-muted);font-size:7.7px}.uad-progress{height:3px;overflow:hidden;border-radius:999px;background:var(--uad-line-soft)}.uad-progress>div{height:100%;width:0;background:#96a3ac;transition:width .15s ease}.uad-empty{padding:22px 12px;text-align:center;color:var(--uad-muted);font-size:8.6px}.uad-ui.is-busy .uad-provider{opacity:.55}
    .uad-assets::-webkit-scrollbar,.uad-url::-webkit-scrollbar,.uad-status::-webkit-scrollbar,.uad-sources::-webkit-scrollbar{width:7px;height:7px}.uad-assets::-webkit-scrollbar-track,.uad-url::-webkit-scrollbar-track,.uad-status::-webkit-scrollbar-track,.uad-sources::-webkit-scrollbar-track{background:transparent}.uad-assets::-webkit-scrollbar-thumb,.uad-url::-webkit-scrollbar-thumb,.uad-status::-webkit-scrollbar-thumb,.uad-sources::-webkit-scrollbar-thumb{background:#465058;border-radius:999px}.uad-assets::-webkit-scrollbar-thumb:hover,.uad-url::-webkit-scrollbar-thumb:hover,.uad-status::-webkit-scrollbar-thumb:hover,.uad-sources::-webkit-scrollbar-thumb:hover{background:#58636c}
    @container (max-width:560px){.uad-provider{display:none}.uad-input-row{grid-template-columns:minmax(0,1fr) auto}.uad-cancel{grid-column:2}.uad-summary{grid-template-columns:repeat(2,minmax(0,1fr))}.uad-asset{grid-template-columns:auto minmax(0,1fr)}.uad-destination{grid-column:2}}
  `;
  document.head.appendChild(style);
}

function iconize(element, icon, label) {
  if (!element) return;
  element.innerHTML = `${icon}<span>${label}</span>`;
}

function findWidget(node, name) {
  return (node.widgets || []).find((widget) => widget?.name === name) || null;
}

function hideNativeWidget(widget) {
  if (!widget) return;
  widget.__uadHardHidden = true;
  widget.hidden = true;
  widget.type = "uad-hidden";
  widget.options = { ...(widget.options || {}), hidden: true };
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
}

function hardHideLegacyWidgets(node) {
  for (const widget of node.widgets || []) {
    if (LEGACY_WIDGETS.has(widget?.name)) hideNativeWidget(widget);
  }
}

function bytesLabel(bytes) {
  let value = Number(bytes) || 0;
  if (value <= 0) return "Unknown";
  let unit = "B";
  for (const candidate of ["B", "KB", "MB", "GB", "TB"]) {
    unit = candidate;
    if (value < 1024 || candidate === "TB") break;
    value /= 1024;
  }
  return `${value.toFixed(unit === "GB" || unit === "TB" ? 2 : 1)} ${unit}`;
}

function providerLabel(url) {
  const value = String(url || "").toLowerCase();
  if (value.includes("huggingface.co")) return "Hugging Face";
  if (value.includes("civitai.com")) return "Civitai";
  if (value.includes("mega.nz")) return "MEGA";
  return "Model link";
}

function extractUrls(text) {
  const matches = String(text || "").match(/https?:\/\/.*?(?=https?:\/\/|\s|$)/gi) || [];
  const seen = new Set();
  const urls = [];
  for (let match of matches) {
    match = match.trim().replace(/[),;\]}]+$/g, "");
    if (!match || seen.has(match)) continue;
    seen.add(match);
    urls.push(match);
  }
  return urls;
}

function assetKey(asset) {
  return [asset?.provider, asset?.repo_id, asset?.revision, asset?.remote_path, asset?.filename]
    .map((value) => String(value || ""))
    .join("|");
}

function targetLabel(asset) {
  return `models/${asset.destination || "unclassified"}/${asset.filename || "model"}`;
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

async function mapLimit(items, limit, worker, signal, onProgress) {
  const results = new Array(items.length);
  let cursor = 0;
  let completed = 0;

  async function run() {
    while (true) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
      const index = cursor;
      cursor += 1;
      if (index >= items.length) return;
      try {
        results[index] = { ok: true, value: await worker(items[index], index) };
      } catch (error) {
        if (error?.name === "AbortError") throw error;
        results[index] = { ok: false, error };
      }
      completed += 1;
      onProgress?.(completed, items.length);
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => run()));
  return results;
}

function installDownloaderUI(node) {
  if (node.__uadAuthoritativeUI || typeof node.addDOMWidget !== "function") return;
  node.__uadAuthoritativeUI = true;
  installStyles();

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
  root.className = "uad-ui";
  root.innerHTML = `
    <div class="uad-head">
      <div><div class="uad-title">🌐 Universal Asset Downloader</div><div class="uad-subtitle">Analyze first. UAD classifies every discovered model file and routes it to the matching ComfyUI folder.</div></div>
      <div class="uad-provider">Paste model links</div>
    </div>
    <div class="uad-input-row">
      <textarea class="uad-url" spellcheck="false" placeholder="One model URL per line — Hugging Face, Civitai, or MEGA"></textarea>
      <button class="uad-btn uad-btn-primary uad-analyze" type="button"></button>
      <button class="uad-btn uad-cancel" type="button" hidden></button>
    </div>
    <div class="uad-input-help"><strong>Batch:</strong><span class="uad-input-summary">one URL per line · duplicates ignored · up to ${MAX_SOURCES} sources</span></div>
    <div class="uad-toolbar">
      <button class="uad-btn uad-select-all" type="button" disabled></button>
      <button class="uad-btn uad-clear" type="button" disabled></button>
      <span class="uad-spacer"></span>
    </div>
    <div class="uad-summary" hidden>
      <div class="uad-stat"><div class="uad-stat-label">Sources</div><div class="uad-stat-value uad-source-count">0</div></div>
      <div class="uad-stat"><div class="uad-stat-label">Files found</div><div class="uad-stat-value uad-count">0</div></div>
      <div class="uad-stat"><div class="uad-stat-label">Combined size</div><div class="uad-stat-value uad-total">Unknown</div></div>
      <div class="uad-stat"><div class="uad-stat-label">Selected</div><div class="uad-stat-value uad-selected">Nothing selected</div></div>
    </div>
    <div class="uad-sources" hidden></div>
    <div class="uad-assets"><div class="uad-empty">Paste one or more model URLs, then Analyze. Analysis never downloads model files.</div></div>
    <div class="uad-notice" hidden></div>
    <details class="uad-details">
      <summary>${ICONS.settings}<span>Access tokens & advanced</span></summary>
      <div class="uad-secret-grid">
        <div class="uad-field"><label>Hugging Face token (optional)</label><input class="uad-hf-token" type="password" autocomplete="off"></div>
        <div class="uad-field"><label>Civitai API key (optional)</label><input class="uad-civitai-token" type="password" autocomplete="off"></div>
      </div>
      <label class="uad-force"><input type="checkbox"> Force replace an existing file only when you intentionally want to redownload it</label>
    </details>
    <div class="uad-toolbar">
      <button class="uad-btn uad-verify" type="button" disabled></button>
      <span class="uad-spacer"></span>
      <button class="uad-btn uad-btn-primary uad-install" type="button" disabled></button>
    </div>
    <div class="uad-status">Ready. Analysis is metadata-only.</div>
    <div class="uad-progress"><div></div></div>
  `;

  const el = {
    provider: root.querySelector(".uad-provider"),
    url: root.querySelector(".uad-url"),
    inputSummary: root.querySelector(".uad-input-summary"),
    analyze: root.querySelector(".uad-analyze"),
    cancel: root.querySelector(".uad-cancel"),
    selectAll: root.querySelector(".uad-select-all"),
    clear: root.querySelector(".uad-clear"),
    summary: root.querySelector(".uad-summary"),
    sourceCount: root.querySelector(".uad-source-count"),
    count: root.querySelector(".uad-count"),
    total: root.querySelector(".uad-total"),
    selected: root.querySelector(".uad-selected"),
    sources: root.querySelector(".uad-sources"),
    assets: root.querySelector(".uad-assets"),
    notice: root.querySelector(".uad-notice"),
    hf: root.querySelector(".uad-hf-token"),
    civitai: root.querySelector(".uad-civitai-token"),
    force: root.querySelector(".uad-force input"),
    verify: root.querySelector(".uad-verify"),
    install: root.querySelector(".uad-install"),
    status: root.querySelector(".uad-status"),
    progress: root.querySelector(".uad-progress > div"),
  };

  iconize(el.analyze, ICONS.analyze, "Analyze");
  iconize(el.cancel, ICONS.cancel, "Cancel");
  iconize(el.selectAll, ICONS.selectAll, "Select all");
  iconize(el.clear, ICONS.clear, "Clear");
  iconize(el.verify, ICONS.verify, "Verify");
  iconize(el.install, ICONS.install, "Install selected");

  const state = {
    assets: [],
    selected: new Set(),
    busy: false,
    controller: null,
    sources: [],
    sourceFailures: [],
  };

  el.url.value = native.url?.value || "";
  el.hf.value = native.hf?.value || "";
  el.civitai.value = native.civitai?.value || "";
  el.force.checked = Boolean(native.force?.value);

  const selectedAssets = () => state.assets.filter((asset) => state.selected.has(asset.id));

  const syncNative = () => {
    if (native.url) native.url.value = el.url.value.trim();
    if (native.hf) native.hf.value = el.hf.value;
    if (native.civitai) native.civitai.value = el.civitai.value;
    if (native.force) native.force.value = Boolean(el.force.checked);
    if (native.type) native.type.value = "Auto";
    if (native.selection) native.selection.value = JSON.stringify(selectedAssets());
  };

  const updateInputSummary = () => {
    const urls = extractUrls(el.url.value);
    const providers = [...new Set(urls.map(providerLabel))];
    el.provider.textContent = urls.length > 1 ? `${urls.length} links · ${providers.join(" + ")}` : urls.length === 1 ? providers[0] : "Paste model links";
    el.inputSummary.textContent = urls.length
      ? `${urls.length} source${urls.length === 1 ? "" : "s"} · ${providers.join(" + ")} · duplicates ignored · max ${MAX_SOURCES}`
      : `one URL per line · duplicates ignored · up to ${MAX_SOURCES} sources`;
    return urls;
  };

  const updateSelection = () => {
    const chosen = selectedAssets();
    const bytes = chosen.reduce((sum, asset) => sum + (Number(asset.size_bytes) || 0), 0);
    const unsupported = chosen.filter((asset) => !["huggingface", "civitai"].includes(String(asset.provider || "").toLowerCase()));
    el.selected.textContent = chosen.length ? `${chosen.length} · ${bytesLabel(bytes)}` : "Nothing selected";
    el.install.disabled = state.busy || !chosen.length || chosen.length > MAX_INSTALL_ITEMS || unsupported.length > 0;
    el.verify.disabled = state.busy || !chosen.length;
    el.selectAll.disabled = state.busy || !state.assets.length;
    el.clear.disabled = state.busy || !state.assets.length;
    if (unsupported.length && !state.busy) {
      el.install.title = `${unsupported.length} selected item${unsupported.length === 1 ? "" : "s"} use a review-only provider. Hugging Face and Civitai are installable.`;
    } else if (chosen.length > MAX_INSTALL_ITEMS) {
      el.install.title = `Select at most ${MAX_INSTALL_ITEMS} files per install.`;
    } else {
      el.install.title = "Install selected model files";
    }
    syncNative();
  };

  const setBusy = (busy, message = "") => {
    state.busy = busy;
    root.classList.toggle("is-busy", busy);
    el.analyze.disabled = busy;
    el.cancel.hidden = !busy;
    updateSelection();
    if (message) el.status.textContent = message;
  };

  const destinationSelect = (asset) => {
    const select = document.createElement("select");
    select.className = "uad-destination";
    for (const [value, label] of DESTINATIONS) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = `models/${value}`;
      option.title = label;
      option.selected = value === asset.destination;
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
      empty.textContent = "No supported model files found.";
      el.assets.appendChild(empty);
      updateSelection();
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
        if (check.checked) state.selected.add(asset.id);
        else state.selected.delete(asset.id);
        updateSelection();
      });

      const file = document.createElement("div");
      file.className = "uad-file";
      const link = document.createElement("a");
      link.className = "uad-file-name";
      link.href = asset.source_url || asset.__sourceUrl || "#";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = asset.remote_path || asset.filename;
      link.title = link.textContent;

      const meta = document.createElement("div");
      meta.className = "uad-meta";
      const confidence = Number(asset.confidence) || 0;
      const confidenceClass = confidence >= 0.85 ? "uad-chip-high" : confidence >= 0.55 ? "uad-chip-medium" : "uad-chip-low";
      const confidenceText = confidence >= 0.85 ? "high confidence" : confidence >= 0.55 ? "review" : "manual review";
      const chips = [
        `${asset.provider_label || providerLabel(asset.__sourceUrl)} · ${asset.__sourceTitle || "source"}`,
        asset.size_label || bytesLabel(asset.size_bytes),
        asset.asset_type || "Unknown",
        confidenceText,
        asset.sha256 ? "SHA256 available" : "basic verify",
      ];
      chips.forEach((text, index) => {
        const chip = document.createElement("span");
        chip.className = `uad-chip${index === 3 ? ` ${confidenceClass}` : ""}`;
        chip.textContent = text;
        if (index === 3) chip.title = String(asset.reason || "");
        meta.appendChild(chip);
      });

      const target = document.createElement("div");
      target.className = "uad-target";
      target.innerHTML = `${ICONS.file}<span></span>`;
      target.querySelector("span").textContent = targetLabel(asset);
      target.title = `Expected ComfyUI path: ${targetLabel(asset)}`;
      file.append(link, meta, target);
      row.append(check, file, destinationSelect(asset));
      el.assets.appendChild(row);
    }
    updateSelection();
  };

  const renderSources = () => {
    el.sources.replaceChildren();
    for (const source of state.sources) {
      const link = document.createElement("a");
      link.className = "uad-source-link";
      link.href = source.sourceUrl || source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.innerHTML = ICONS.external;
      const label = document.createElement("span");
      label.textContent = `${source.providerLabel} · ${source.title}`;
      link.appendChild(label);
      el.sources.appendChild(link);
    }
    for (const failure of state.sourceFailures) {
      const item = document.createElement("span");
      item.className = "uad-source-link is-error";
      item.innerHTML = ICONS.link;
      const label = document.createElement("span");
      label.textContent = `${providerLabel(failure.url)} failed`;
      item.appendChild(label);
      item.title = String(failure.error?.message || failure.error || "Analysis failed");
      el.sources.appendChild(item);
    }
    el.sources.hidden = !el.sources.childElementCount;
  };

  const resetAnalysisSurface = () => {
    state.assets = [];
    state.selected.clear();
    state.sources = [];
    state.sourceFailures = [];
    el.summary.hidden = true;
    el.sources.hidden = true;
    el.notice.hidden = true;
    el.progress.style.width = "0%";
    renderAssets();
  };

  const analyze = async () => {
    const urls = updateInputSummary();
    if (!urls.length) {
      el.status.textContent = "Paste at least one model URL first.";
      return;
    }
    if (urls.length > MAX_SOURCES) {
      el.status.textContent = `Too many links. Analyze at most ${MAX_SOURCES} sources at once.`;
      return;
    }

    state.controller?.abort();
    const controller = new AbortController();
    state.controller = controller;
    syncNative();
    setBusy(true, `Analyzing ${urls.length} source${urls.length === 1 ? "" : "s"} · up to ${ANALYZE_CONCURRENCY} at once · ComfyUI stays usable.`);
    el.progress.style.width = "6%";

    try {
      const settled = await mapLimit(
        urls,
        ANALYZE_CONCURRENCY,
        (url) => postJson("/uad/analyze-fast", {
          url,
          hf_token: el.hf.value,
          civitai_api_key: el.civitai.value,
        }, controller.signal),
        controller.signal,
        (done, total) => {
          el.progress.style.width = `${Math.max(8, Math.round((done / total) * 92))}%`;
          el.status.textContent = `Analyzing source ${done}/${total}…`;
        },
      );
      if (controller.signal.aborted) return;

      const assets = [];
      const seenAssets = new Set();
      const sources = [];
      const failures = [];
      const notices = [];

      settled.forEach((entry, index) => {
        const url = urls[index];
        if (!entry?.ok) {
          failures.push({ url, error: entry?.error });
          return;
        }
        const result = entry.value;
        const sourceIndex = sources.length;
        const source = {
          url,
          sourceUrl: result.source_url || url,
          provider: result.provider || "",
          providerLabel: result.provider_label || providerLabel(url),
          title: String(result.title || result.provider_label || providerLabel(url)),
          originalIndex: index,
        };
        sources.push(source);
        if (result.notice) notices.push(`${source.title}: ${result.notice}`);

        for (const rawAsset of Array.isArray(result.assets) ? result.assets : []) {
          const key = assetKey(rawAsset);
          if (seenAssets.has(key)) continue;
          seenAssets.add(key);
          const id = String(rawAsset.id || `${key}|${index}`);
          assets.push({
            ...rawAsset,
            id,
            __sourceUrl: source.sourceUrl,
            __sourceTitle: source.title,
            __sourceIndex: sourceIndex,
            __originalSourceIndex: index,
          });
        }
      });

      if (!sources.length) {
        throw new Error(failures.map((failure) => `${providerLabel(failure.url)}: ${failure.error?.message || failure.error}`).join("\n") || "No source could be analyzed.");
      }

      state.assets = assets;
      state.sources = sources;
      state.sourceFailures = failures;
      state.selected.clear();

      for (const sourceIndex of [...new Set(assets.map((asset) => asset.__sourceIndex))]) {
        const group = assets.filter((asset) => asset.__sourceIndex === sourceIndex);
        if (group.length === 1) state.selected.add(group[0].id);
        else group.filter((asset) => asset.primary).forEach((asset) => state.selected.add(asset.id));
      }

      const totalBytes = assets.reduce((sum, asset) => sum + (Number(asset.size_bytes) || 0), 0);
      el.summary.hidden = false;
      el.sourceCount.textContent = failures.length ? `${sources.length}/${urls.length}` : String(sources.length);
      el.count.textContent = String(assets.length);
      el.total.textContent = bytesLabel(totalBytes);

      const failureNote = failures.length
        ? `${failures.length} source${failures.length === 1 ? "" : "s"} failed: ${failures.map((failure) => `${providerLabel(failure.url)} · ${failure.error?.message || failure.error}`).join(" | ")}`
        : "";
      const megaNote = assets.some((asset) => String(asset.provider || "").toLowerCase() === "mega")
        ? "MEGA is analysis/review-only in the safe installer; Hugging Face and Civitai assets can be installed directly."
        : "";
      el.notice.textContent = [...new Set(notices), failureNote, megaNote].filter(Boolean).join("\n");
      el.notice.hidden = !el.notice.textContent;

      renderSources();
      renderAssets();
      el.progress.style.width = "100%";
      el.status.textContent = `${sources.length}/${urls.length} source${urls.length === 1 ? "" : "s"} analyzed · ${assets.length} model file${assets.length === 1 ? "" : "s"} found${failures.length ? ` · ${failures.length} failed` : ""}.`;
      setTimeout(() => {
        if (!state.busy) el.progress.style.width = "0%";
      }, 650);
    } catch (error) {
      if (error?.name === "AbortError") {
        el.status.textContent = "Analysis canceled.";
      } else {
        resetAnalysisSurface();
        el.assets.innerHTML = '<div class="uad-empty">Analysis failed. Check the URLs or access tokens and try again.</div>';
        el.status.textContent = `Analysis failed: ${error.message}`;
      }
    } finally {
      if (state.controller === controller) state.controller = null;
      setBusy(false);
    }
  };

  el.url.addEventListener("input", () => {
    updateInputSummary();
    if (native.url) native.url.value = el.url.value;
  });
  el.hf.addEventListener("input", syncNative);
  el.civitai.addEventListener("input", syncNative);
  el.force.addEventListener("change", syncNative);

  el.analyze.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    analyze();
  });
  el.cancel.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    state.controller?.abort();
  });
  el.selectAll.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    state.assets.forEach((asset) => state.selected.add(asset.id));
    renderAssets();
  });
  el.clear.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    state.selected.clear();
    renderAssets();
  });

  el.verify.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    const items = selectedAssets();
    if (!items.length || state.busy) return;
    setBusy(true, `Verifying ${items.length} local file${items.length === 1 ? "" : "s"} in a worker thread…`);
    try {
      const result = await postJson("/uad/verify-fast", { items });
      const rows = result.results || [];
      const good = rows.filter((item) => item.ok).length;
      const details = rows.map((item, index) => {
        const path = item.path || targetLabel(items[index] || {});
        return `${item.ok ? "OK" : "CHECK"} · ${path}\n${item.message || ""}`;
      }).join("\n\n");
      el.status.textContent = `${good}/${rows.length} verified.${details ? `\n${details}` : ""}`;
    } catch (error) {
      el.status.textContent = `Verification failed: ${error.message}`;
    } finally {
      setBusy(false);
    }
  });

  el.install.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    const items = selectedAssets();
    if (!items.length || state.busy) return;
    const unsupported = items.filter((asset) => !["huggingface", "civitai"].includes(String(asset.provider || "").toLowerCase()));
    if (unsupported.length) {
      el.status.textContent = "Selected MEGA/review-only assets cannot be installed by the safe downloader yet.";
      return;
    }
    if (items.length > MAX_INSTALL_ITEMS) {
      el.status.textContent = `Select at most ${MAX_INSTALL_ITEMS} files per install.`;
      return;
    }

    syncNative();
    setBusy(true, `Installing ${items.length} selected asset${items.length === 1 ? "" : "s"}…`);
    el.progress.style.width = "2%";
    try {
      const result = await postJson("/uad/install", {
        items,
        node_id: String(node.id ?? ""),
        hf_token: el.hf.value,
        civitai_api_key: el.civitai.value,
        force: Boolean(el.force.checked),
      });
      const rows = result.results || [];
      const installed = rows.filter((item) => item.ok && !item.skipped).length;
      const reused = rows.filter((item) => item.ok && item.skipped).length;
      el.progress.style.width = "100%";
      el.status.textContent = `Install complete · ${installed} installed${reused ? ` · ${reused} already verified` : ""}.`;
      setTimeout(() => {
        if (!state.busy) el.progress.style.width = "0%";
      }, 800);
    } catch (error) {
      el.progress.style.width = "0%";
      el.status.textContent = `Install failed: ${error.message}`;
    } finally {
      setBusy(false);
    }
  });

  node.__uadUI = { root, el, state, renderAssets };
  const widget = node.addDOMWidget("uad_ui", "uad_ui", root, { serialize: false, hideOnZoom: false });
  widget.computeSize = (width) => [Math.max(0, width), Math.max(470, (node.size?.[1] || 660) - 86)];
  node.setSize?.([Math.max(node.size?.[0] || 700, 700), Math.max(node.size?.[1] || 660, 660)]);
  hardHideLegacyWidgets(node);
  updateInputSummary();
  updateSelection();
}

api.addEventListener("uad-progress", ({ detail }) => {
  const node = app.graph?.getNodeById?.(Number(detail?.node)) || app.graph?.getNodeById?.(detail?.node);
  const ui = node?.__uadUI;
  if (!ui) return;
  if (detail?.status) ui.el.status.textContent = detail.status;
  if (Number.isFinite(Number(detail?.progress))) {
    ui.el.progress.style.width = `${Math.max(0, Math.min(100, Number(detail.progress)))}%`;
  }
});

app.registerExtension({
  name: "Comfy.UniversalAssetDownloader.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET) return;
    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function uadExecuted(message) {
      originalExecuted?.apply(this, arguments);
      const text = message?.text?.[0] || message?.download_message?.[0];
      if (this.__uadUI && text) this.__uadUI.el.status.textContent = text;
    };
  },
  async nodeCreated(node) {
    if (node.comfyClass === TARGET || node.type === TARGET) installDownloaderUI(node);
  },
});
