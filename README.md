<div align="center">

# Universal Asset Downloader

### Install ComfyUI models without guessing what they are or where they belong.

Paste a Hugging Face or Civitai link. UAD inspects provider metadata first, classifies the asset, shows its size and destination, downloads it safely, then verifies the exact file that landed in `ComfyUI/models`.

<p>
  <a href="https://github.com/thaakeno/comfyui-universal-asset-downloader/stargazers"><img alt="Star Universal Asset Downloader" src="https://img.shields.io/badge/%E2%98%85%20Star-UAD-22C55E?style=for-the-badge&logo=github&logoColor=white&labelColor=171B1F"></a>
  <img alt="ComfyUI custom node" src="https://img.shields.io/badge/ComfyUI-Custom%20Node-0EA5E9?style=for-the-badge&labelColor=171B1F">
  <img alt="Hugging Face Xet" src="https://img.shields.io/badge/Hugging%20Face-hf__xet-FFD21E?style=for-the-badge&logo=huggingface&logoColor=111827&labelColor=171B1F">
  <img alt="Verified installs" src="https://img.shields.io/badge/Install-Verified%20%2B%20Atomic-A855F7?style=for-the-badge&labelColor=171B1F">
  <img alt="Version 2.1.1" src="https://img.shields.io/badge/Version-2.1.1-F59E0B?style=for-the-badge&labelColor=171B1F">
  <a href="#license"><img alt="AGPLv3 license" src="https://img.shields.io/badge/License-AGPLv3-22C55E?style=for-the-badge&labelColor=171B1F"></a>
</p>

**Analyze first. Download fast. Put the file in the right place. Verify what arrived.**

</div>

> [!IMPORTANT]
> UAD is intentionally conservative. If provider metadata is not strong enough to identify an asset safely, it goes to `models/unclassified` for review instead of being silently guessed as a checkpoint.

## Why UAD

Model links are rarely just “a file.” A Hugging Face repository can contain diffusion models, LoRAs, VAEs, text encoders, preview VAEs, PDD heads and several alternate checkpoints at once. Civitai links can point at a model, a specific version, or an image recipe.

UAD turns those links into an install plan before touching disk. It resolves provider metadata, detects the model role, previews the exact `ComfyUI/models/...` path, checks available space, downloads into temporary staging, validates size/hash/format, and only then promotes the file into its final location.

## What you get

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>Fast Hugging Face Xet downloads</h3>
      Hugging Face installs use <code>huggingface_hub</code> + <code>hf_xet</code> instead of a plain streamed HTTP loop. Xet adaptive concurrency is available everywhere; UAD enables <code>HF_XET_HIGH_PERFORMANCE=1</code> for its known Lightning <code>/teamspace</code> profile and on machines with at least 64 GiB RAM, while respecting explicit overrides.
    </td>
    <td width="50%" valign="top">
      <h3>Metadata-aware routing</h3>
      LoRAs, VAEs, text encoders, CLIP vision models, diffusion models, ControlNets, upscalers, embeddings, checkpoints, preview VAEs and H3 PDD heads are routed separately instead of being dumped into one folder.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>Verified, atomic installs</h3>
      Known provider SHA256 and file sizes are checked when available. New files are staged first; a forced repair does not replace the existing file until the replacement has already passed verification.
    </td>
    <td width="50%" valign="top">
      <h3>Useful live progress</h3>
      External integrations such as H3 Studio receive current filename, overall progress, downloaded/total bytes and file X/Y. Large multi-file installs no longer look frozen while the backend is working.
    </td>
  </tr>
</table>

## Hugging Face download engine

UAD 2.1 replaces its old Hugging Face `requests.get(..., stream=True)` transfer path with the Hub's native download stack:

```text
Hugging Face metadata
        ↓
huggingface_hub.hf_hub_download
        ↓
hf_xet / Xet adaptive concurrency
        ↓
same-filesystem UAD staging
        ↓
size + SHA256 + format verification
        ↓
atomic promotion into ComfyUI/models/...
```

No shell command is required. UAD does **not** need to invoke `wget` or spawn the `hf` CLI; `hf_hub_download` uses `hf_xet` directly when it is installed. This keeps the backend portable across Windows, Linux, local workstations and cloud machines.

UAD keeps a persistent Hub/Xet cache under the active ComfyUI installation unless you already provide Hugging Face cache environment variables. It also removes the deprecated `HF_HUB_ENABLE_HF_TRANSFER` setting and uses the current Xet path instead.

### High-performance policy

The default is automatic:

- an existing `HF_XET_HIGH_PERFORMANCE` value is respected;
- `UAD_HF_XET_HIGH_PERFORMANCE=1` forces HP mode;
- `UAD_HF_XET_HIGH_PERFORMANCE=0` forces normal adaptive mode;
- Lightning `/teamspace` installs use the high-bandwidth profile used by H3 Studio;
- other machines with at least 64 GiB RAM enable HP mode automatically;
- everything else keeps Xet adaptive concurrency.

This means UAD stays fast on high-bandwidth machines without making Lightning-specific paths a requirement for normal use.

## Supported sources

### Hugging Face

Best-supported source. Paste a repository, folder (`/tree/...`) or direct file (`/blob/...` or `/resolve/...`) URL. Public repositories need no token; gated/private assets can use a Hugging Face token from the node's Advanced section.

Repository analysis shows every supported model file instead of blindly installing the whole repo.

### Civitai

Paste a model URL, a URL containing `modelVersionId`, a direct model-version download URL, or a Civitai image URL with recipe resources. An API key is optional unless the asset requires one.

### MEGA

MEGA links can be recognized, but automatic smart install remains intentionally disabled because the provider does not expose enough trustworthy metadata for the same destination/hash guarantees.

## Install

### ComfyUI Manager / Registry

Search for **Universal Asset Downloader** in ComfyUI Manager / Extensions and install the latest Registry version.

### Git

```bash
git clone https://github.com/thaakeno/comfyui-universal-asset-downloader.git
cd comfyui-universal-asset-downloader
python -m pip install -r requirements.txt
```

Restart ComfyUI and hard-refresh the browser.

## Use

Add **Universal Asset Downloader** from `utilities/downloaders`.

1. Paste a supported link.
2. Click **Analyze**. This fetches metadata only; it does not install the model.
3. Review the discovered files, sizes, confidence and destinations.
4. Select only what you actually need.
5. Click **Install selected**.
6. Use **Verify installed** whenever you want to re-check local files.

The downloader writes only underneath the active ComfyUI `models` directory.

## H3 Studio integration

UAD exposes a small backend integration API so another custom node can request the same safe install path without reimplementing download logic. MiniMax H3 Studio uses this for its Model Setup panel, including matched PDD LoRA + heads installation.

The split is deliberate:

```text
H3 Studio → knows which H3 assets belong together
UAD       → owns provider download, path safety, progress and verification
```

PDD head artifacts use `models/pdd_heads`; their matching student LoRAs use `models/loras`.

## Detection philosophy

Automatic routing should be useful without being reckless. High-confidence signatures and provider declarations are routed automatically. Ambiguous files are shown as **Unclassified / review** rather than silently becoming checkpoints.

Putting an unknown multi-gigabyte asset in the wrong ComfyUI directory is worse than asking for one deliberate confirmation.

## Security and verification

UAD accepts installs only from supported HTTPS provider hosts. Destination paths are normalized and constrained underneath `ComfyUI/models`. Known-size assets get a free-space preflight. Existing files are verified before they are skipped. Provider SHA256 is checked when available, and supported model formats receive basic structural validation.

Hugging Face repairs use same-filesystem staging so a verified replacement can be atomically promoted with `os.replace`. Tokens are only sent to their matching provider and are not written into normal download logs.

## Development

```bash
python -m pytest -q
python -m compileall __init__.py nodes
```

The Registry package version lives in `pyproject.toml`. Updating that version on `main` automatically runs the Comfy Registry publishing workflow.

## License

Universal Asset Downloader is available under the [GNU AGPLv3](LICENSE).
