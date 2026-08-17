import { app } from "/scripts/app.js";

const STYLE_ID = "uad-compact-theme-v1";

function installTheme() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    html body .uad-ui{
      --uad-control:color-mix(in srgb,var(--comfy-input-bg,#181c20) 94%,#080b0e 6%);
      --uad-control-hover:color-mix(in srgb,var(--comfy-input-bg,#181c20) 86%,white 8%);
      --uad-control-active:color-mix(in srgb,var(--uad-control) 88%,#526577 12%);
      --uad-line:color-mix(in srgb,var(--border-color,#41484f) 72%,transparent);
      --uad-line-soft:color-mix(in srgb,var(--border-color,#41484f) 43%,transparent);
      --uad-muted:color-mix(in srgb,var(--descrip-text,#7f8992) 94%,white 6%);
      --uad-text:var(--input-text,var(--fg-color,#e7eaed));
      width:100%!important;height:100%!important;min-width:0!important;min-height:0!important;
      padding:8px 10px 10px!important;gap:7px!important;overflow:hidden!important;
      background:transparent!important;background-image:none!important;border:0!important;border-radius:0!important;box-shadow:none!important;
      color:var(--uad-text)!important;font:10.5px/1.4 Inter,ui-sans-serif,system-ui,sans-serif!important;
    }
    html body .uad-ui .uad-head{margin:0!important;align-items:flex-start!important;gap:12px!important;padding:1px 1px 6px!important;border-bottom:1px solid var(--uad-line-soft)!important}
    html body .uad-ui .uad-title{font-size:13px!important;line-height:1.2!important;font-weight:720!important;color:#edf0f2!important}
    html body .uad-ui .uad-subtitle{font-size:9px!important;color:var(--uad-muted)!important;margin-top:3px!important;max-width:480px!important}
    html body .uad-ui .uad-provider{flex:0 0 auto!important;max-width:180px!important;min-height:27px!important;padding:4px 8px!important;border:1px solid var(--uad-line)!important;border-radius:7px!important;background:var(--uad-control)!important;color:#b2bbc1!important;font-size:9px!important;box-shadow:none!important}

    html body .uad-ui .uad-input-row{grid-template-columns:minmax(0,1fr) auto auto!important;gap:6px!important;min-width:0!important;align-items:stretch!important}
    html body .uad-ui .uad-url{width:100%!important;height:58px!important;min-height:58px!important;max-height:58px!important;resize:none!important;overflow:auto!important;padding:8px 9px!important;border:1px solid var(--uad-line)!important;border-radius:7px!important;background:var(--uad-control)!important;color:var(--uad-text)!important;box-shadow:none!important;font:9.7px/1.38 ui-monospace,SFMono-Regular,Consolas,monospace!important;white-space:pre!important}
    html body .uad-ui .uad-url:hover{background:var(--uad-control-hover)!important}
    html body .uad-ui .uad-url:focus{background:var(--uad-control-hover)!important;border-color:#59636c!important;box-shadow:0 0 0 1px color-mix(in srgb,#59636c 42%,transparent)!important;outline:none!important}
    html body .uad-ui .uad-input-help{min-height:16px!important;color:var(--uad-muted)!important;font-size:8px!important}
    html body .uad-ui .uad-input-help strong{color:#aeb7bd!important}

    html body .uad-ui .uad-btn,
    html body .uad-ui .uad-source-link{min-height:29px!important;padding:6px 9px!important;border:1px solid var(--uad-line)!important;border-radius:7px!important;background:var(--uad-control)!important;color:#cbd1d6!important;box-shadow:none!important;font:650 9.5px/1 Inter,ui-sans-serif,system-ui,sans-serif!important}
    html body .uad-ui .uad-btn:hover:not(:disabled),
    html body .uad-ui .uad-source-link:hover{background:var(--uad-control-hover)!important;border-color:#59636c!important;color:#f0f2f4!important}
    html body .uad-ui .uad-btn-primary{background:var(--uad-control-active)!important;border-color:#56636e!important;color:#edf0f2!important}
    html body .uad-ui .uad-btn-primary:hover:not(:disabled){background:var(--uad-control-hover)!important;border-color:#64717b!important}
    html body .uad-ui .uad-cancel{border-color:color-mix(in srgb,#9f6464 42%,var(--uad-line))!important;background:color-mix(in srgb,var(--uad-control) 92%,#7d4141 8%)!important}

    html body .uad-ui .uad-toolbar{margin:0!important;gap:5px!important;min-width:0!important}
    html body .uad-ui .uad-summary{margin:0!important;gap:0!important;border-top:1px solid var(--uad-line-soft)!important;border-bottom:1px solid var(--uad-line-soft)!important;background:transparent!important}
    html body .uad-ui .uad-stat{padding:6px 8px!important;border:0!important;border-right:1px solid var(--uad-line-soft)!important;border-radius:0!important;background:transparent!important;box-shadow:none!important}
    html body .uad-ui .uad-stat:last-child{border-right:0!important}
    html body .uad-ui .uad-stat-label{font-size:7.5px!important;letter-spacing:.075em!important;color:var(--uad-muted)!important}
    html body .uad-ui .uad-stat-value{font-size:9.5px!important;color:#dce0e3!important}
    html body .uad-ui .uad-sources{gap:5px!important;padding:0 0 1px!important}
    html body .uad-ui .uad-source-link{min-height:25px!important;padding:4px 7px!important;color:#9aa4ab!important;font-size:8px!important}

    html body .uad-ui .uad-assets{flex:1 1 auto!important;min-height:82px!important;max-height:none!important;overflow:auto!important;gap:5px!important;padding:1px 2px 1px 0!important;scrollbar-gutter:stable!important}
    html body .uad-ui .uad-asset{grid-template-columns:auto minmax(0,1fr) minmax(116px,148px)!important;gap:8px!important;padding:7px 8px!important;border:1px solid var(--uad-line)!important;border-radius:8px!important;background:transparent!important;box-shadow:none!important}
    html body .uad-ui .uad-asset:hover{background:color-mix(in srgb,var(--uad-control) 18%,transparent)!important}
    html body .uad-ui .uad-file-name{font-size:10px!important;color:#e3e6e8!important}
    html body .uad-ui .uad-meta{gap:3px!important;margin-top:4px!important}
    html body .uad-ui .uad-chip{min-height:0!important;font-size:7.5px!important;padding:1px 5px!important;border:1px solid var(--uad-line-soft)!important;border-radius:5px!important;background:transparent!important;color:#9ea8af!important}
    html body .uad-ui .uad-destination{width:100%!important;min-width:0!important;max-width:none!important;font-size:8.5px!important;padding:4px 5px!important;border:1px solid var(--uad-line)!important;border-radius:6px!important;background:var(--uad-control)!important;color:#cfd4d8!important;box-shadow:none!important}
    html body .uad-ui .uad-destination:hover,
    html body .uad-ui .uad-destination:focus{background:var(--uad-control-hover)!important;border-color:#59636c!important}
    html body .uad-ui .uad-target{margin-top:5px!important;color:#7f8991!important;font:8.5px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace!important}

    html body .uad-ui .uad-notice{margin:0!important;padding:6px 8px!important;border:1px solid color-mix(in srgb,#b18b45 34%,var(--uad-line))!important;border-radius:7px!important;background:color-mix(in srgb,#b18b45 7%,transparent)!important;color:#b9ad94!important;font-size:8.5px!important}
    html body .uad-ui .uad-details{margin:0!important;border:1px solid var(--uad-line)!important;border-radius:7px!important;overflow:hidden!important;background:transparent!important;box-shadow:none!important}
    html body .uad-ui .uad-details summary{padding:6px 8px!important;font-size:9px!important;color:#aeb6bc!important;background:transparent!important}
    html body .uad-ui .uad-details summary:hover{background:color-mix(in srgb,var(--uad-control) 20%,transparent)!important}
    html body .uad-ui .uad-field input{padding:5px 6px!important;font-size:8.5px!important;border:1px solid var(--uad-line)!important;background:var(--uad-control)!important;color:var(--uad-text)!important;box-shadow:none!important}
    html body .uad-ui .uad-status{margin:0!important;min-height:29px!important;max-height:58px!important;padding:6px 8px!important;border:1px solid var(--uad-line-soft)!important;border-radius:7px!important;background:transparent!important;color:#aeb6bc!important;box-shadow:none!important;font-size:8.5px!important}
    html body .uad-ui .uad-progress{margin:0!important;height:3px!important;background:color-mix(in srgb,var(--border-color,#41484f) 52%,transparent)!important}
    html body .uad-ui .uad-progress>div{background:#9aa6ad!important;opacity:.72!important}
    html body .uad-ui .uad-empty{min-height:74px!important;padding:14px!important;font-size:9px!important;color:var(--uad-muted)!important}

    html body .uad-ui .uad-assets::-webkit-scrollbar,
    html body .uad-ui .uad-url::-webkit-scrollbar,
    html body .uad-ui .uad-status::-webkit-scrollbar,
    html body .uad-ui .uad-sources::-webkit-scrollbar{width:7px;height:7px}
    html body .uad-ui .uad-assets::-webkit-scrollbar-track,
    html body .uad-ui .uad-url::-webkit-scrollbar-track,
    html body .uad-ui .uad-status::-webkit-scrollbar-track,
    html body .uad-ui .uad-sources::-webkit-scrollbar-track{background:transparent}
    html body .uad-ui .uad-assets::-webkit-scrollbar-thumb,
    html body .uad-ui .uad-url::-webkit-scrollbar-thumb,
    html body .uad-ui .uad-status::-webkit-scrollbar-thumb,
    html body .uad-ui .uad-sources::-webkit-scrollbar-thumb{background:color-mix(in srgb,var(--border-color,#41484f) 74%,transparent);border-radius:99px}

    @container (max-width:560px){
      html body .uad-ui .uad-provider{display:none!important}
      html body .uad-ui .uad-input-row{grid-template-columns:minmax(0,1fr) auto!important}
      html body .uad-ui .uad-cancel{grid-column:2!important}
      html body .uad-ui .uad-summary{grid-template-columns:repeat(2,minmax(0,1fr))!important}
      html body .uad-ui .uad-asset{grid-template-columns:auto minmax(0,1fr)!important}
      html body .uad-ui .uad-destination{grid-column:2!important}
    }
  `;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "Comfy.UniversalAssetDownloader.Theme",
  setup() {
    installTheme();
  },
});
