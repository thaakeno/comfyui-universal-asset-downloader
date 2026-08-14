import { app } from "/scripts/app.js";

const TARGET = "UniversalAssetDownloader";

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

function selectedAssets(ui) {
  return ui.state.assets.filter((asset) => ui.state.selected.has(asset.id));
}

function targetLabel(asset) {
  return `models/${asset.destination || "unclassified"}/${asset.filename || "model"}`;
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
      target.style.cssText = "margin-top:5px;opacity:.68;font-size:10px;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Consolas,monospace";
      file.appendChild(target);
    }
    target.textContent = `→ ${targetLabel(asset)}`;
    target.title = `Expected ComfyUI path: ${targetLabel(asset)}`;
  });
}

function installPolish(node, attempt = 0) {
  if (node.__uadV2PolishInstalled) return;
  const ui = node.__uadV2;
  if (!ui) {
    if (attempt < 20) setTimeout(() => installPolish(node, attempt + 1), 25);
    return;
  }
  node.__uadV2PolishInstalled = true;

  patchRows(ui);
  const observer = new MutationObserver(() => patchRows(ui));
  observer.observe(ui.el.assets, { childList: true, subtree: true });
  node.__uadV2PolishObserver = observer;

  // Replace the legacy compact verification message with an exact-path report.
  ui.el.verify.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();

    const items = selectedAssets(ui);
    if (!items.length || ui.state.busy) return;
    ui.state.busy = true;
    ui.el.verify.disabled = true;
    ui.el.install.disabled = true;
    ui.el.status.textContent = `Verifying ${items.length} local file${items.length === 1 ? "" : "s"}…`;
    try {
      const result = await postJson("/uad/verify", { items });
      const results = result.results || [];
      const good = results.filter((item) => item.ok).length;
      const details = results.map((item, index) => {
        const expected = targetLabel(items[index] || {});
        const path = item.path || expected;
        return `${item.ok ? "✓" : "✗"} ${item.status || "unknown"}\n  ${path}\n  ${item.message || ""}`;
      }).join("\n");
      ui.el.status.textContent = `${good}/${results.length} verified.\n${details}`;
    } catch (error) {
      ui.el.status.textContent = `Verification failed: ${error.message}`;
    } finally {
      ui.state.busy = false;
      const hasSelection = selectedAssets(ui).length > 0;
      ui.el.verify.disabled = !hasSelection;
      ui.el.install.disabled = !hasSelection;
    }
  }, true);
}

app.registerExtension({
  name: "Comfy.UniversalAssetDownloader.v2.polish",
  async nodeCreated(node) {
    if (node.comfyClass === TARGET || node.type === TARGET) installPolish(node);
  },
});
