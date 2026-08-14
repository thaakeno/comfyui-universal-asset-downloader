# Universal Asset Downloader for ComfyUI

A model installer that analyzes a link before it downloads anything, works out what the file actually is, shows how large it is, chooses the appropriate `ComfyUI/models/...` destination, and verifies the local file after installation.

## What v2 changes

The old downloader mostly inferred a model type from a filename and could fall back to `checkpoints` when it was unsure. v2 is deliberately stricter.

- **Analyze before download.** Hugging Face and Civitai metadata is inspected first.
- **Smart model routing.** LoRAs, VAEs, text encoders, CLIP vision models, diffusion models/UNets, ControlNets, upscalers, embeddings and checkpoints have separate detection rules.
- **Safe ambiguity handling.** Unknown `.safetensors` files are shown as **Unclassified / review** instead of silently becoming checkpoints.
- **Hugging Face repository browser.** Paste a repo, folder (`/tree/...`) or direct file (`/blob/...` or `/resolve/...`) link and see the matching model files.
- **Civitai model + image recipe support.** Model links expose files from the selected/latest version; image links reconstruct model resources from recipe metadata when available.
- **Per-file sizes and totals.** The UI shows every discovered file, repository total and selected download size.
- **Destination preview + override.** See `models/loras`, `models/vae`, `models/text_encoders`, `models/diffusion_models`, etc. before installing and override a destination when you know better.
- **Verification button.** Provider SHA256 is used when Hugging Face LFS or Civitai exposes it. Otherwise v2 performs size/basic format checks.
- **Atomic downloads.** Files download to `.part` first and are only moved into the model folder after the transfer completes.
- **Disk-space preflight.** Known-size assets are checked against free space before downloading.
- **Existing-file safety.** A local file is verified before it is skipped. A mismatching file is not silently overwritten unless you explicitly enable Force replace.
- **No arbitrary download hosts.** Install selections are restricted to the supported provider hosts.
- **Better frontend.** Provider icon, source/file hyperlinks, file cards, confidence labels, destination picker, selected/total GB, live progress, Analyze, Install selected and Verify installed.

## Supported sources

### Hugging Face

Best supported. Paste a model repository, folder or direct model-file URL. Public repositories do not need a token. A Hugging Face token can be entered in the node's Advanced section for gated/private models.

### Civitai

Paste a model URL, a URL with a `modelVersionId`, a direct model-version download URL, or a Civitai image URL containing recipe resources. An API key is optional unless the asset requires one.

### MEGA

MEGA links can still be recognized, but v2 intentionally does **not** perform an automatic smart install from MEGA yet because MEGA links do not expose enough trustworthy model metadata for the same destination/hash guarantees. The UI tells you this instead of pretending the file is a checkpoint.

## Installation

Clone into `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/thaakeno/comfyui-universal-asset-downloader.git
```

Then install the tiny runtime dependency in the same Python environment ComfyUI uses:

```bash
python -m pip install -r comfyui-universal-asset-downloader/requirements.txt
```

Restart ComfyUI and hard-refresh the browser.

If you install through ComfyUI Manager/Registry, the dependency installation is normally handled for you.

## Usage

Add **Universal Asset Downloader** from `utilities/downloaders`.

1. Paste a Hugging Face or Civitai link.
2. Click **Analyze**. This is metadata-only and does not download the model.
3. Review every detected file, size, type, confidence and destination.
4. Select the files you actually want. Multi-file repositories are not blindly installed in full.
5. Click **Install selected**.
6. Click **Verify installed** whenever you want to re-check the local files.

The downloader writes only underneath the active ComfyUI `models` directory.

## Detection philosophy

Automatic routing should be useful without being reckless. High-confidence signatures such as `lora`, `lightx2v`, `vae`, `text_encoder`, `qwen3vl`, `controlnet`, `clip_vision`, known diffusion-model families, and provider-declared Civitai model types are routed automatically. If metadata is insufficient, v2 chooses `models/unclassified` and marks the file for manual review.

This is intentional: putting an unknown multi-GB model in the wrong ComfyUI folder is worse than asking for one click of confirmation.

## Security / verification

Downloads are written to a temporary `.part` file. Known SHA256 values are checked before the final file is accepted. Existing files are verified before being treated as already installed. Verification endpoints cannot hash arbitrary machine paths; they are restricted to `ComfyUI/models`.

Tokens are used only for requests to their matching provider and are not printed into download logs.

## License

GNU AGPLv3. See [LICENSE](LICENSE).
