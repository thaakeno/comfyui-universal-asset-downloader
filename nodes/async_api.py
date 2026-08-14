from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import quote

import requests
from aiohttp import web
from server import PromptServer

from . import smart_asset_service as service

_ORIGINAL_HF_ANALYZER = service.analyze_huggingface
_DIRECT_TIMEOUT = (5, 15)


def _hf_headers(token: str) -> dict[str, str]:
    headers = {"User-Agent": service.USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _header_size(headers: requests.structures.CaseInsensitiveDict[str]) -> int | None:
    for key in ("x-linked-size", "x-repo-size"):
        value = headers.get(key)
        if value:
            try:
                size = int(value)
                if size > 0:
                    return size
            except (TypeError, ValueError):
                pass
    return None


def _header_sha256(headers: requests.structures.CaseInsensitiveDict[str]) -> str:
    for key in ("x-linked-etag", "etag"):
        value = str(headers.get(key) or "").strip().strip('"').removeprefix("W/").strip('"')
        value = value.removeprefix("sha256:")
        if re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return value.lower()
    return ""


def analyze_huggingface_fast(url: str, token: str = "") -> dict[str, Any]:
    """Resolve direct HF files without enumerating the whole repository.

    Repository/tree URLs still use the metadata API, but callers of the async
    endpoint execute that work in a worker thread so ComfyUI's aiohttp loop is
    never blocked.
    """

    repo_id, revision, direct_file, _prefix = service._hf_parse(url)
    if not direct_file:
        return _ORIGINAL_HF_ANALYZER(url, token)

    headers = _hf_headers(token)
    resolve_url = (
        f"https://huggingface.co/{repo_id}/resolve/{quote(revision, safe='')}/"
        f"{quote(direct_file, safe='/')}?download=true"
    )
    size: int | None = None
    sha256 = ""
    warning = ""

    try:
        response = requests.head(
            resolve_url,
            headers=headers,
            allow_redirects=False,
            timeout=_DIRECT_TIMEOUT,
        )
        if response.status_code in {401, 403, 404}:
            response.raise_for_status()
        if response.status_code >= 400 and response.status_code not in {405}:
            response.raise_for_status()
        size = _header_size(response.headers)
        sha256 = _header_sha256(response.headers)

        if size is None and response.is_redirect:
            target = response.headers.get("location")
            if target:
                final = requests.head(
                    target,
                    headers={"User-Agent": service.USER_AGENT},
                    allow_redirects=True,
                    timeout=_DIRECT_TIMEOUT,
                )
                final.raise_for_status()
                content_length = final.headers.get("content-length")
                if content_length:
                    try:
                        size = int(content_length)
                    except (TypeError, ValueError):
                        pass
                sha256 = sha256 or _header_sha256(final.headers)
    except requests.RequestException as exc:
        # Keep direct-file installs usable even when the provider does not
        # expose HEAD metadata. Download/verification still performs its own
        # safety checks.
        warning = f"Provider metadata was unavailable: {exc}"

    inferred = service.infer_destination(repo_id, direct_file)
    asset = {
        "id": f"hf:{repo_id}:{revision}:{direct_file}",
        "provider": "huggingface",
        "provider_label": "Hugging Face",
        "repo_id": repo_id,
        "revision": revision,
        "remote_path": direct_file,
        "filename": service._safe_filename(direct_file),
        "size_bytes": size,
        "size_label": service.human_size(size),
        "sha256": sha256,
        "download_url": resolve_url,
        "source_url": f"https://huggingface.co/{repo_id}/blob/{revision}/{quote(direct_file, safe='/')}",
        **inferred,
    }
    notice = "Direct file metadata resolved without scanning the full Hugging Face repository."
    if warning:
        notice += f" {warning}"
    return {
        "provider": "huggingface",
        "provider_label": "Hugging Face",
        "title": repo_id,
        "source_url": url,
        "assets": [asset],
        "total_bytes": int(size or 0),
        "total_size_label": service.human_size(size),
        "notice": notice,
    }


# The original /uad/analyze route resolves this module global at request time,
# so direct-file links also become much cheaper for older frontends.
service.analyze_huggingface = analyze_huggingface_fast


def _verify_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        destination = item.get("destination") or "unclassified"
        path = service.safe_target(destination, item.get("filename") or "")
        output.append(service.verify_file(path, item.get("size_bytes"), item.get("sha256") or ""))
    return output


@PromptServer.instance.routes.post("/uad/analyze-fast")
async def api_analyze_fast(request):
    try:
        payload = await request.json()
        result = await asyncio.to_thread(
            service.analyze_url,
            payload.get("url", ""),
            payload.get("hf_token", ""),
            payload.get("civitai_api_key", ""),
        )
        return web.json_response({"ok": True, **result})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


@PromptServer.instance.routes.post("/uad/verify-fast")
async def api_verify_fast(request):
    try:
        payload = await request.json()
        items = payload.get("items") or []
        if not isinstance(items, list):
            raise ValueError("Verification items must be a list.")
        results = await asyncio.to_thread(_verify_items, items)
        return web.json_response({"ok": True, "results": results})
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


__all__ = ["analyze_huggingface_fast"]
