from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
from aiohttp import web
from server import PromptServer

try:
    import folder_paths
except Exception:  # pragma: no cover - ComfyUI provides this at runtime
    folder_paths = None

MODEL_EXTENSIONS = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".onnx"}
ALLOWED_DESTINATIONS = {
    "checkpoints",
    "loras",
    "vae",
    "controlnet",
    "upscale_models",
    "text_encoders",
    "clip",
    "clip_vision",
    "diffusion_models",
    "unet",
    "embeddings",
    "style_models",
    "audio_encoders",
    "unclassified",
}
USER_AGENT = "ComfyUI-Universal-Asset-Downloader/2"
REQUEST_TIMEOUT = (15, 60)
DOWNLOAD_TIMEOUT = (20, 180)


def models_dir() -> Path:
    if folder_paths is not None:
        candidate = getattr(folder_paths, "models_dir", None)
        if candidate:
            return Path(candidate).expanduser().resolve()
    # .../ComfyUI/custom_nodes/comfyui-universal-asset-downloader/nodes/this_file.py
    return Path(__file__).resolve().parents[3] / "models"


def human_size(size: int | None) -> str:
    if not size:
        return "Unknown"
    value = float(size)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit in {"GB", "TB"} else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def _safe_filename(value: str) -> str:
    value = unquote(str(value or "")).replace("\\", "/")
    name = value.rsplit("/", 1)[-1].strip().replace("\x00", "")
    name = re.sub(r"[\r\n\t]", "_", name)
    if not name or name in {".", ".."}:
        raise ValueError("The remote asset does not have a safe filename.")
    return name


def safe_target(destination: str, filename: str) -> Path:
    destination = str(destination or "").strip().lower()
    if destination not in ALLOWED_DESTINATIONS:
        raise ValueError(f"Unsupported destination: {destination!r}")
    root = models_dir()
    target = (root / destination / _safe_filename(filename)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Refusing to write outside ComfyUI/models.") from exc
    return target


def _contains(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def infer_destination(repo_id: str, filename: str, declared_type: str = "", tags: list[str] | None = None) -> dict[str, Any]:
    declared = (declared_type or "").strip().lower()
    joined_tags = " ".join(tags or []).lower()
    text = f"{repo_id} {filename} {joined_tags}".lower().replace("-", "_")
    suffix = Path(filename).suffix.lower()

    if declared in {"lora", "lycoris", "locon"} or _contains(text, "lora", "lycoris", "locon", "lightx2v") or re.search(r"minimax_h3_.*turbo_\d+step", text):
        return {"asset_type": "LoRA", "destination": "loras", "confidence": 0.99, "reason": "LoRA/adapter metadata or filename signature"}
    if declared == "vae" or _contains(text, "_vae", "/vae", "video_vae", "image_vae", "taeh", "autoencoder"):
        return {"asset_type": "VAE", "destination": "vae", "confidence": 0.98, "reason": "VAE/autoencoder signature"}
    if declared in {"controlnet", "control net"} or _contains(text, "controlnet", "control_net", "t2i_adapter"):
        return {"asset_type": "ControlNet", "destination": "controlnet", "confidence": 0.98, "reason": "ControlNet signature"}
    if declared in {"upscaler", "upscale model"} or _contains(text, "upscaler", "realesrgan", "esrgan", "swinir", "ultrasharp"):
        return {"asset_type": "Upscale Model", "destination": "upscale_models", "confidence": 0.98, "reason": "upscaler signature"}
    if _contains(text, "clip_vision", "siglip", "vision_encoder"):
        return {"asset_type": "CLIP Vision", "destination": "clip_vision", "confidence": 0.95, "reason": "vision encoder signature"}
    if _contains(text, "text_encoder", "text_encoders", "qwen3vl", "qwen3_vl", "qwen2vl", "qwen2_vl", "umt5", "t5xxl", "t5_xxl", "clip_l", "clip_g", "gemma_text"):
        return {"asset_type": "Text Encoder", "destination": "text_encoders", "confidence": 0.96, "reason": "text-encoder family signature"}
    if _contains(text, "audio_encoder", "audio_vae", "vocoder"):
        return {"asset_type": "Audio Model", "destination": "audio_encoders", "confidence": 0.88, "reason": "audio model signature"}
    if declared in {"textualinversion", "embedding", "embeddings"} or _contains(text, "textual_inversion", "embedding"):
        return {"asset_type": "Embedding", "destination": "embeddings", "confidence": 0.94, "reason": "embedding signature"}
    if _contains(text, "style_model", "style_models"):
        return {"asset_type": "Style Model", "destination": "style_models", "confidence": 0.90, "reason": "style model signature"}
    if declared == "checkpoint":
        return {"asset_type": "Checkpoint", "destination": "checkpoints", "confidence": 0.96, "reason": "provider declares a checkpoint"}
    if _contains(text, "diffusion_model", "diffusion_models", "transformer", "unet", "flux", "wan2", "wan_2", "hunyuan", "ltx", "cosmos", "mochi", "minimax_h3", "hidream", "sd3_medium"):
        return {"asset_type": "Diffusion Model", "destination": "diffusion_models", "confidence": 0.90, "reason": "diffusion-transformer/UNet family signature"}
    if suffix == ".ckpt" or _contains(text, "checkpoint", "sdxl_base", "sd15", "sd_1_5"):
        return {"asset_type": "Checkpoint", "destination": "checkpoints", "confidence": 0.84, "reason": "checkpoint filename/extension signature"}
    if suffix == ".gguf":
        return {"asset_type": "Diffusion Model", "destination": "diffusion_models", "confidence": 0.68, "reason": "GGUF model with no stronger role signature"}
    return {"asset_type": "Unclassified", "destination": "unclassified", "confidence": 0.25, "reason": "not enough trustworthy metadata; manual review recommended"}


def _hf_parse(url: str) -> tuple[str, str, str | None, str | None]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError("Hugging Face URL must include owner/repository.")
    repo_id = f"{parts[0]}/{parts[1]}"
    revision = "main"
    file_path = None
    prefix = None
    if len(parts) >= 4 and parts[2] in {"blob", "resolve", "tree"}:
        revision = parts[3]
        remainder = "/".join(parts[4:]) or None
        if parts[2] in {"blob", "resolve"}:
            file_path = remainder
        else:
            prefix = remainder
    return repo_id, revision, file_path, prefix


def _hf_api_info(repo_id: str, revision: str, token: str) -> dict[str, Any]:
    quoted_repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    url = f"https://huggingface.co/api/models/{quoted_repo}/revision/{quote(revision, safe='')}?blobs=true"
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if response.status_code == 404 and revision == "main":
        response = requests.get(f"https://huggingface.co/api/models/{quoted_repo}?blobs=true", headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _hf_asset(repo_id: str, revision: str, sibling: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    path = sibling.get("rfilename") or sibling.get("path") or ""
    lfs = sibling.get("lfs") or {}
    size = sibling.get("size") or lfs.get("size")
    oid = lfs.get("oid") or ""
    sha256 = oid.removeprefix("sha256:") if str(oid).startswith("sha256:") else ""
    inferred = infer_destination(repo_id, path, tags=tags)
    return {
        "id": f"hf:{repo_id}:{revision}:{path}",
        "provider": "huggingface",
        "provider_label": "Hugging Face",
        "repo_id": repo_id,
        "revision": revision,
        "remote_path": path,
        "filename": _safe_filename(path),
        "size_bytes": int(size) if size else None,
        "size_label": human_size(int(size)) if size else "Unknown",
        "sha256": sha256.lower(),
        "download_url": f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(path, safe='/')}?download=true",
        "source_url": f"https://huggingface.co/{repo_id}/blob/{revision}/{quote(path, safe='/')}",
        **inferred,
    }


def analyze_huggingface(url: str, token: str = "") -> dict[str, Any]:
    repo_id, revision, direct_file, prefix = _hf_parse(url)
    info = _hf_api_info(repo_id, revision, token)
    tags = [str(tag) for tag in info.get("tags", [])]
    siblings = info.get("siblings", []) or []
    candidates = []
    for sibling in siblings:
        path = sibling.get("rfilename") or sibling.get("path") or ""
        if direct_file and path != direct_file:
            continue
        if prefix and not path.startswith(prefix.rstrip("/") + "/") and path != prefix:
            continue
        if Path(path).suffix.lower() not in MODEL_EXTENSIONS:
            continue
        candidates.append(_hf_asset(repo_id, revision, sibling, tags))
    if direct_file and not candidates:
        # Some private/LFS metadata endpoints omit size/hash; keep the direct file usable.
        inferred = infer_destination(repo_id, direct_file, tags=tags)
        candidates.append({
            "id": f"hf:{repo_id}:{revision}:{direct_file}",
            "provider": "huggingface",
            "provider_label": "Hugging Face",
            "repo_id": repo_id,
            "revision": revision,
            "remote_path": direct_file,
            "filename": _safe_filename(direct_file),
            "size_bytes": None,
            "size_label": "Unknown",
            "sha256": "",
            "download_url": f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(direct_file, safe='/')}?download=true",
            "source_url": url,
            **inferred,
        })
    if not candidates:
        raise ValueError("No supported model files were found at this Hugging Face URL.")
    candidates.sort(key=lambda item: (-float(item["confidence"]), item["remote_path"]))
    return {
        "provider": "huggingface",
        "provider_label": "Hugging Face",
        "title": repo_id,
        "source_url": url,
        "assets": candidates,
        "total_bytes": sum(int(item["size_bytes"] or 0) for item in candidates),
        "total_size_label": human_size(sum(int(item["size_bytes"] or 0) for item in candidates)),
        "notice": "Review multi-file repositories before installing. Ambiguous files are quarantined in models/unclassified instead of being guessed as checkpoints.",
    }


def _civitai_headers(api_key: str) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _civitai_version_assets(version: dict[str, Any], source_url: str, declared_type: str = "") -> list[dict[str, Any]]:
    model = version.get("model") or {}
    model_type = declared_type or model.get("type") or ""
    model_name = model.get("name") or version.get("name") or "Civitai model"
    version_id = version.get("id") or "unknown"
    output = []
    for index, file_info in enumerate(version.get("files", []) or []):
        filename = file_info.get("name") or f"civitai-{version_id}-{index}.safetensors"
        hashes = {str(k).lower(): str(v) for k, v in (file_info.get("hashes") or {}).items()}
        sha256 = hashes.get("sha256", "").lower()
        size_kb = file_info.get("sizeKB")
        size = int(float(size_kb) * 1024) if size_kb else None
        inferred = infer_destination(model_name, filename, declared_type=model_type)
        output.append({
            "id": f"civitai:{version_id}:{file_info.get('id', index)}",
            "provider": "civitai",
            "provider_label": "Civitai",
            "repo_id": str(model.get("id") or ""),
            "revision": str(version_id),
            "remote_path": filename,
            "filename": _safe_filename(filename),
            "size_bytes": size,
            "size_label": human_size(size),
            "sha256": sha256,
            "download_url": file_info.get("downloadUrl") or f"https://civitai.com/api/download/models/{version_id}",
            "source_url": source_url,
            "primary": bool(file_info.get("primary")),
            **inferred,
        })
    return output


def _civitai_version(version_id: str, api_key: str) -> dict[str, Any]:
    response = requests.get(
        f"https://civitai.com/api/v1/model-versions/{version_id}",
        headers=_civitai_headers(api_key),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def analyze_civitai(url: str, api_key: str = "") -> dict[str, Any]:
    parsed = urlparse(url)
    image_match = re.search(r"/images/(\d+)", parsed.path)
    if image_match:
        image_id = image_match.group(1)
        response = requests.get(
            f"https://civitai.com/api/v1/images?imageId={image_id}&nsfw=X",
            headers=_civitai_headers(api_key),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items:
            raise ValueError("Civitai did not return metadata for that image.")
        meta = items[0].get("meta") or {}
        version_ids: list[str] = []
        for resource in meta.get("civitaiResources", []) or []:
            version_id = resource.get("modelVersionId")
            if version_id:
                version_ids.append(str(version_id))
        for resource in meta.get("additionalResources", []) or []:
            match = re.search(r"civitai:\d+@(\d+)", str(resource.get("name") or ""))
            if match:
                version_ids.append(match.group(1))
        assets: list[dict[str, Any]] = []
        for version_id in dict.fromkeys(version_ids):
            assets.extend(_civitai_version_assets(_civitai_version(version_id, api_key), url))
        if not assets:
            raise ValueError("No downloadable model resources were found in the Civitai image metadata.")
        total = sum(int(item["size_bytes"] or 0) for item in assets)
        return {"provider": "civitai", "provider_label": "Civitai", "title": f"Civitai image {image_id}", "source_url": url, "assets": assets, "total_bytes": total, "total_size_label": human_size(total), "notice": "Assets were reconstructed from the image recipe metadata."}

    direct_version = re.search(r"/api/download/models/(\d+)", parsed.path)
    if direct_version:
        version = _civitai_version(direct_version.group(1), api_key)
        assets = _civitai_version_assets(version, url)
    else:
        model_match = re.search(r"/models/(\d+)", parsed.path)
        if not model_match:
            raise ValueError("Could not determine the Civitai model ID.")
        model_id = model_match.group(1)
        response = requests.get(
            f"https://civitai.com/api/v1/models/{model_id}",
            headers=_civitai_headers(api_key),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        model = response.json()
        versions = model.get("modelVersions", []) or []
        requested = (parse_qs(parsed.query).get("modelVersionId") or [None])[0]
        version = next((item for item in versions if str(item.get("id")) == str(requested)), None) if requested else None
        version = version or (versions[0] if versions else None)
        if not version:
            raise ValueError("Civitai model has no downloadable versions.")
        version = {**version, "model": {**(version.get("model") or {}), "id": model.get("id"), "name": model.get("name"), "type": model.get("type")}}
        assets = _civitai_version_assets(version, url, model.get("type") or "")
    if not assets:
        raise ValueError("Civitai model version contains no downloadable files.")
    assets.sort(key=lambda item: (not item.get("primary", False), -float(item["confidence"]), item["filename"]))
    total = sum(int(item["size_bytes"] or 0) for item in assets)
    return {"provider": "civitai", "provider_label": "Civitai", "title": "Civitai model", "source_url": url, "assets": assets, "total_bytes": total, "total_size_label": human_size(total), "notice": "Primary files are listed first. Review alternate precision/pruning files before selecting more than one."}


def analyze_mega(url: str) -> dict[str, Any]:
    filename = "mega_asset"
    match = re.search(r"([^/?#]+\.(?:safetensors|gguf|ckpt|pt|pth|bin))", url, re.I)
    if match:
        filename = match.group(1)
    inferred = infer_destination("mega", filename)
    asset = {
        "id": f"mega:{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
        "provider": "mega",
        "provider_label": "MEGA",
        "repo_id": "",
        "revision": "",
        "remote_path": filename,
        "filename": filename,
        "size_bytes": None,
        "size_label": "Unknown",
        "sha256": "",
        "download_url": url,
        "source_url": url,
        **inferred,
    }
    return {"provider": "mega", "provider_label": "MEGA", "title": "MEGA asset", "source_url": url, "assets": [asset], "total_bytes": 0, "total_size_label": "Unknown", "notice": "MEGA does not expose enough trusted metadata for automatic size/hash verification. Confirm the destination before installing."}


def analyze_url(url: str, hf_token: str = "", civitai_api_key: str = "") -> dict[str, Any]:
    url = str(url or "").strip()
    if not url:
        raise ValueError("Paste an asset URL first.")
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":", 1)[0]
    if host in {"huggingface.co", "www.huggingface.co"}:
        return analyze_huggingface(url, hf_token)
    if host in {"civitai.com", "www.civitai.com"}:
        return analyze_civitai(url, civitai_api_key)
    if host in {"mega.nz", "www.mega.nz"}:
        return analyze_mega(url)
    raise ValueError("Unsupported source. Use a Hugging Face, Civitai, or MEGA URL.")


def _quick_format_check(path: Path) -> tuple[bool, str]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            with path.open("rb") as handle:
                raw = handle.read(8)
                if len(raw) != 8:
                    return False, "Safetensors header is truncated."
                header_len = struct.unpack("<Q", raw)[0]
                if header_len <= 1 or header_len > 100 * 1024 * 1024:
                    return False, "Safetensors header length is invalid."
                header = json.loads(handle.read(header_len).decode("utf-8"))
                if not isinstance(header, dict) or not header:
                    return False, "Safetensors header is empty or invalid."
            return True, "Safetensors header is valid."
        except Exception as exc:
            return False, f"Safetensors validation failed: {exc}"
    if suffix == ".gguf":
        try:
            with path.open("rb") as handle:
                return (handle.read(4) == b"GGUF", "GGUF magic header checked.")
        except Exception as exc:
            return False, f"GGUF validation failed: {exc}"
    return True, "File exists and basic checks passed."


def verify_file(path: Path, expected_size: int | None = None, expected_sha256: str = "") -> dict[str, Any]:
    root = models_dir()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Verification is restricted to ComfyUI/models.") from exc
    if not resolved.is_file():
        return {"ok": False, "status": "missing", "message": "File is not installed.", "path": str(resolved)}
    actual_size = resolved.stat().st_size
    if expected_size and actual_size != int(expected_size):
        return {"ok": False, "status": "size_mismatch", "message": f"Size mismatch: expected {human_size(int(expected_size))}, found {human_size(actual_size)}.", "path": str(resolved), "size_bytes": actual_size}
    expected_sha256 = str(expected_sha256 or "").lower().removeprefix("sha256:")
    digest = ""
    if expected_sha256:
        hasher = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected_sha256:
            return {"ok": False, "status": "hash_mismatch", "message": "SHA256 mismatch. The local file is not the expected artifact.", "path": str(resolved), "sha256": digest, "size_bytes": actual_size}
    format_ok, format_message = _quick_format_check(resolved)
    return {"ok": bool(format_ok), "status": "verified" if format_ok else "invalid", "message": format_message if format_ok else format_message, "path": str(resolved), "sha256": digest or expected_sha256, "size_bytes": actual_size}


def _download_http(asset: dict[str, Any], hf_token: str, civitai_api_key: str, force: bool, progress_callback=None) -> dict[str, Any]:
    destination = asset.get("destination") or "unclassified"
    target = safe_target(destination, asset.get("filename") or "")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(asset.get("size_bytes") or 0) or None
    expected_hash = str(asset.get("sha256") or "").lower()

    if target.exists() and not force:
        verification = verify_file(target, expected_size, expected_hash)
        if verification["ok"]:
            return {**verification, "skipped": True, "asset": asset}
        raise ValueError(f"{target.name} already exists but failed verification. Enable force download to replace it safely.")

    if expected_size:
        free = shutil.disk_usage(target.parent).free
        required = expected_size + max(512 * 1024 * 1024, int(expected_size * 0.03))
        if free < required:
            raise OSError(f"Not enough free disk space for {target.name}: need about {human_size(required)}, have {human_size(free)}.")

    headers = {"User-Agent": USER_AGENT}
    if asset.get("provider") == "huggingface" and hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    if asset.get("provider") == "civitai" and civitai_api_key:
        headers["Authorization"] = f"Bearer {civitai_api_key}"

    temp = target.with_suffix(target.suffix + ".part")
    temp.unlink(missing_ok=True)
    hasher = hashlib.sha256()
    downloaded = 0
    try:
        with requests.get(asset["download_url"], headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            response_size = int(response.headers.get("content-length") or 0) or expected_size
            with temp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress_callback and response_size:
                        progress_callback(downloaded, int(response_size), asset)
                handle.flush()
                os.fsync(handle.fileno())
        if expected_size and downloaded != expected_size:
            raise IOError(f"Downloaded size mismatch for {target.name}: expected {expected_size} bytes, got {downloaded}.")
        digest = hasher.hexdigest()
        if expected_hash and digest != expected_hash:
            raise IOError(f"SHA256 mismatch for {target.name}; download was discarded.")
        format_ok, format_message = _quick_format_check(temp)
        if not format_ok:
            raise IOError(format_message)
        os.replace(temp, target)
        verification = verify_file(target, expected_size, expected_hash)
        return {**verification, "skipped": False, "asset": asset}
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def download_assets(assets: list[dict[str, Any]], hf_token: str = "", civitai_api_key: str = "", force: bool = False, progress_callback=None) -> list[dict[str, Any]]:
    results = []
    for asset in assets:
        if asset.get("provider") == "mega":
            raise ValueError("MEGA smart installs require a manual destination and are not enabled in the safe multi-file installer yet.")
        results.append(_download_http(asset, hf_token, civitai_api_key, force, progress_callback))
    return results


@PromptServer.instance.routes.post("/uad/analyze")
async def api_analyze(request):
    try:
        payload = await request.json()
        result = analyze_url(
            payload.get("url", ""),
            hf_token=payload.get("hf_token", ""),
            civitai_api_key=payload.get("civitai_api_key", ""),
        )
        return web.json_response({"ok": True, **result})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/uad/verify")
async def api_verify(request):
    try:
        payload = await request.json()
        items = payload.get("items") or []
        output = []
        for item in items:
            destination = item.get("destination") or "unclassified"
            path = safe_target(destination, item.get("filename") or "")
            output.append(verify_file(path, item.get("size_bytes"), item.get("sha256") or ""))
        return web.json_response({"ok": True, "results": output})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
