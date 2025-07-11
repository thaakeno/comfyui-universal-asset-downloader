
import os
import re
from urllib.parse import urlparse
import subprocess
import shutil
from typing import Optional, Dict, Any, Tuple
import requests
import tempfile
import json
from server import PromptServer
import importlib.util

# --- BEGIN: Civitai Image/Recipe Import Logic ---

CIVITAI_IMAGE_URL_PATTERNS = [
    r"civitai.com/images/(\d+)",
    r"civitai.com/api/v1/images/(\d+)"
]

# Helper to detect if a URL is a Civitai image URL
def is_civitai_image_url(url):
    for pattern in CIVITAI_IMAGE_URL_PATTERNS:
        if re.search(pattern, url):
            return True
    return False

# Helper to extract image ID from Civitai image URL
def extract_civitai_image_id(url):
    for pattern in CIVITAI_IMAGE_URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Helper to fetch Civitai image metadata (returns dict)
def fetch_civitai_image_metadata(image_id, api_key=""):
    api_url = f"https://civitai.com/api/v1/images?imageId={image_id}&nsfw=X"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(api_url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if data and "items" in data and len(data["items"]) > 0:
        return data["items"][0]
    else:
        raise Exception(f"No image metadata found for ID: {image_id}")

# Helper to fetch model version info and get download URL
def fetch_model_version_info(version_id, api_key=""):
    api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(api_url, headers=headers)
    resp.raise_for_status()
    return resp.json()

# Helper to parse LoRAs and models from Civitai image metadata
def parse_civitai_image_assets(image_metadata):
    # Use dictionaries to track unique resources by version_id (as string)
    unique_checkpoints = {}
    unique_loras = {}

    # The resources are in meta.civitaiResources and meta.additionalResources
    meta = image_metadata.get("meta", {})

    # Process civitaiResources first
    civitai_resources = meta.get("civitaiResources", [])
    for res in civitai_resources:
        res_type = res.get("type", "").lower()
        version_id = str(res.get("modelVersionId")) if res.get("modelVersionId") else None

        if res_type == "checkpoint" and version_id:
            unique_checkpoints[version_id] = res
        elif res_type == "lora" and version_id:
            unique_loras[version_id] = res

    # Process additionalResources (these have URN format)
    additional_resources = meta.get("additionalResources", [])
    for res in additional_resources:
        res_type = res.get("type", "").lower()
        if res_type == "lora":
            # Parse URN to get model version ID
            urn = res.get("name", "")
            # Format: "urn:air:sdxl:lora:civitai:432483@481798"
            match = re.search(r'civitai:\d+@(\d+)', urn)
            if match:
                version_id = str(match.group(1))
                if version_id not in unique_loras:
                    # Add version_id to resource for consistency
                    res["modelVersionId"] = version_id
                    unique_loras[version_id] = res

    # Convert dictionaries back to lists
    loras = list(unique_loras.values())
    checkpoint = list(unique_checkpoints.values())[0] if unique_checkpoints else None

    return loras, checkpoint

# Helper to download a file with progress and text status
def download_file_with_status(url, save_path, log_func, headers=None, progress_callback=None):
    log_func(f"Starting download: {url}")
    log_func(f"Target save path: {os.path.abspath(save_path)}")
    try:
        with requests.get(url, stream=True, headers=headers) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(save_path, 'wb') as f:
                downloaded = 0
                last_percent = -1
                for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            percent = int((downloaded * 100) / total)
                            # Only log every 5% or on completion
                            if percent % 5 == 0 and percent != last_percent or percent == 100:
                                log_func(f"Progress: {percent}% ({downloaded}/{total} bytes)")
                                last_percent = percent
                            # Update progress bar if callback provided
                            if progress_callback and total > 0:
                                progress = (downloaded / total) * 100.0
                                progress_callback.set_progress(progress)
        log_func(f"✅ Download completed: {os.path.basename(save_path)}")
        log_func(f"File saved to: {os.path.abspath(save_path)}")
        # Update final progress
        if progress_callback:
            progress_callback.set_progress(100)
        # Double check file exists
        if not os.path.exists(save_path):
            log_func(f"⚠️  File not found after download: {os.path.abspath(save_path)}. Did you break something?")
        return True
    except Exception as e:
        log_func(f"❌ Download failed: {str(e)}")
        return False

# --- END: Civitai Image/Recipe Import Logic ---

def get_base_dir():
    # This function assumes a specific directory structure relative to the script's location
    # A more robust implementation might use ComfyUI's internal path management
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Assumes this file is in .../custom_nodes/your_node_name/nodes/
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        models_dir = os.path.join(base_dir, 'models')
        if os.path.isdir(models_dir):
            return models_dir
    except NameError: # __file__ is not defined in some contexts
        pass
    # Fallback for standard ComfyUI directory
    return os.path.join(os.getcwd(), "ComfyUI", "models")


def get_model_dirs():
    models_dir = get_base_dir()
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    return model_dirs

def detect_colab_and_base_path():
    # Detect if running in Google Colab
    in_colab = False
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    # Try to get SAVE_TO_GDRIVE from globals or environment
    save_to_gdrive = None
    gdrive_base = None
    if in_colab:
        # Try to get from globals (Colab notebook)
        import builtins
        save_to_gdrive = getattr(builtins, 'SAVE_TO_GDRIVE', None)
        gdrive_base = getattr(builtins, 'GDRIVE_BASE', None)
        if save_to_gdrive is None:
            # Try environment variable
            save_to_gdrive = os.environ.get('SAVE_TO_GDRIVE', 'False').lower() == 'true'
        if gdrive_base is None:
            gdrive_base = os.environ.get('GDRIVE_BASE', '/content/drive/MyDrive/ComfyUI')

    # Default paths
    colab_path = '/content/ComfyUI/models'
    gdrive_path = os.path.join(gdrive_base, 'models') if gdrive_base else '/content/drive/MyDrive/ComfyUI/models'
    local_path = get_base_dir()

    if in_colab:
        if save_to_gdrive:
            return gdrive_path
        else:
            return colab_path
    else:
        return local_path

class UniversalAssetDownloader:
    OUTPUT_NODE = True # This makes the node runnable on its own
    """
    A ComfyUI node that intelligently downloads assets from Civitai, Hugging Face, and MEGA
    """

    def __init__(self):
        self.subfolder_map = {
            "Checkpoint": "checkpoints",
            "LoRA": "loras",
            "VAE": "vae",
            "ControlNet": "controlnet",
            "Upscale Model": "upscale_models",
            "CLIP": "clip",
            "UNET": "unet",
            "TextualInversion": "embeddings",
            "Auto": "auto"
        }
        self.civitai_type_map = {
            "LORA": "LoRA",
            "Checkpoint": "Checkpoint",
            "TextualInversion": "TextualInversion",
            "Hypernetwork": "TextualInversion",
            "AestheticGradient": "TextualInversion",
            "VAE": "VAE",
            "Poses": "ControlNet",
            "Controlnet": "ControlNet",
            "Upscaler": "Upscale Model"
        }
        self.node_id = None
        self.status = "Idle"
        self.progress = 0.0

    def set_progress(self, percentage):
        self.update_status(f"Downloading... {percentage:.1f}%", percentage)

    def update_status(self, status_text, progress=None):
        if progress is not None and hasattr(self, 'node_id') and self.node_id:
            PromptServer.instance.send_sync("progress", {
                "node": self.node_id,
                "value": progress,
                "max": 100
            })

    def prepare_download_path(self, local_path, filename):
        # Use dynamic base path
        base_path = detect_colab_and_base_path()
        full_path = os.path.join(base_path, local_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
        return full_path

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node"""
        return {
            "required": {
                "asset_url": ("STRING", {"multiline": True}),
                "asset_type": (["Auto"] + ["Checkpoint", "LoRA", "VAE", "ControlNet", "Upscale Model", "CLIP", "UNET", "TextualInversion"], {"default": "Auto"}),
                "force_download": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "civitai_api_key": ("STRING", {"multiline": False}),
                "hf_token": ("STRING", {"multiline": False}),
                "base_path": ("STRING", {"default": "./"}),
            },
             "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("download_message",)
    FUNCTION = "download_asset" # This enables the play button
    CATEGORY = "loaders"

    def _detect_asset_type(self, model_data, file_info=None):
        # Try to detect asset type from Civitai model metadata and file info
        # Priority: file_info['type'] > model_data['type'] > file extension
        # Returns one of: 'Checkpoint', 'LoRA', 'VAE', 'ControlNet', 'Upscale Model', 'CLIP', 'UNET', 'TextualInversion'
        type_map = {
            'checkpoint': 'Checkpoint',
            'lora': 'LoRA',
            'vae': 'VAE',
            'controlnet': 'ControlNet',
            'upscaler': 'Upscale Model',
            'clip': 'CLIP',
            'unet': 'UNET',
            'textualinversion': 'TextualInversion',
            'hypernetwork': 'TextualInversion',
            'aestheticgradient': 'TextualInversion',
            'poses': 'ControlNet',
        }
        # 1. File info type
        if file_info and 'type' in file_info and file_info['type']:
            t = file_info['type'].lower()
            if t in type_map:
                return type_map[t]
        # 2. Model data type
        if model_data and 'type' in model_data and model_data['type']:
            t = model_data['type'].lower()
            if t in type_map:
                return type_map[t]
        # 3. File extension
        if file_info and 'name' in file_info:
            fname = file_info['name'].lower()
            if 'vae' in fname:
                return 'VAE'
            if 'lora' in fname:
                return 'LoRA'
            if 'controlnet' in fname or 'control' in fname:
                return 'ControlNet'
            if 'clip' in fname:
                return 'CLIP'
            if 'unet' in fname:
                return 'UNET'
            if 'embeddings' in fname or 'textualinversion' in fname:
                return 'TextualInversion'
        # Default fallback
        return 'Checkpoint'

    def download_asset(self, asset_url, asset_type, force_download, node_id, civitai_api_key="", hf_token="", base_path="./"):
        self.node_id = node_id
        self._log_lines = []
        self._saved_files = []
        def log_func(msg):
            # Aggressive, sarcastic, but helpful log output
            line = f"[UniversalAssetDownloader] {msg}"
            print(line)  # Print to stdout for ComfyUI logs
            self._log_lines.append(line)
        if not asset_url or not asset_url.strip():
            return ("❌ No asset URL provided",)
        domain = urlparse(asset_url).netloc.lower()
        # --- NEW: Handle Civitai image URLs ---
        if is_civitai_image_url(asset_url):
            try:
                image_id = extract_civitai_image_id(asset_url)
                if not image_id:
                    return ("❌ Could not extract image ID from Civitai URL. Are you sure you pasted the right link?",)
                log_func("\n" + "="*60)
                log_func("🎨 CIVITAI IMAGE ASSETS ANALYSIS")
                log_func("="*60)
                log_func(f"Fetching Civitai image metadata for image ID {image_id}...")
                meta = fetch_civitai_image_metadata(image_id, civitai_api_key)
                loras, checkpoint = parse_civitai_image_assets(meta)
                # Print checkpoint info
                if checkpoint:
                    version_id = checkpoint.get("modelVersionId")
                    if version_id:
                        try:
                            version_info = fetch_model_version_info(version_id, civitai_api_key)
                            model_name = version_info.get("model", {}).get("name", "Unknown")
                            version_name = version_info.get("name", "Unknown")
                            files = version_info.get("files", [])
                            log_func(f"\n🏗️  CHECKPOINT:")
                            log_func("-" * 40)
                            log_func(f"  Name: {model_name}")
                            log_func(f"  Version: {version_name}")
                            if files:
                                log_func(f"  Download: {files[0].get('downloadUrl','N/A')}")
                            else:
                                log_func(f"  No files found for checkpoint version {version_id}")
                        except Exception as e:
                            log_func(f"  ❌ Error fetching checkpoint info: {e}")
                # Print LoRA info
                if loras:
                    log_func(f"\n🎨 LORAS ({len(loras)}):")
                    log_func("-" * 40)
                    for i, lora in enumerate(loras, 1):
                        version_id = lora.get("modelVersionId")
                        weight = lora.get("weight", lora.get("strength", "N/A"))
                        if version_id:
                            try:
                                version_info = fetch_model_version_info(version_id, civitai_api_key)
                                model_name = version_info.get("model", {}).get("name", "Unknown")
                                version_name = version_info.get("name", "Unknown")
                                files = version_info.get("files", [])
                                log_func(f"  {i}. Name: {model_name}")
                                log_func(f"     Version: {version_name}")
                                log_func(f"     Weight: {weight}")
                                if files:
                                    log_func(f"     Download: {files[0].get('downloadUrl','N/A')}")
                                else:
                                    log_func(f"     No files found for LoRA version {version_id}")
                            except Exception as e:
                                log_func(f"     ❌ Error fetching LoRA info: {e}")
                # Print generation params if available
                meta_dict = meta if isinstance(meta, dict) else {}
                gen_params = []
                important_params = ["prompt", "negativePrompt", "steps", "sampler", "cfgScale", "seed", "baseModel"]
                for param in important_params:
                    if param in meta_dict:
                        value = meta_dict[param]
                        if param == "prompt" and len(str(value)) > 100:
                            value = str(value)[:100] + "..."
                        gen_params.append(f"  {param}: {value}")
                if gen_params:
                    log_func("\n⚙️  GENERATION PARAMETERS")
                    log_func("="*60)
                    for line in gen_params:
                        log_func(line)
                log_func("\n" + "="*60)
                log_func(f"Found {len(loras)} LoRA(s) and {'1' if checkpoint else '0'} checkpoint in image metadata.")
                if checkpoint:
                    version_id = checkpoint.get("modelVersionId")
                    if version_id:
                        try:
                            log_func(f"Fetching checkpoint version info for ID {version_id}...")
                            version_info = fetch_model_version_info(version_id, civitai_api_key)
                            model_name = version_info.get("model", {}).get("name", "Unknown")
                            version_name = version_info.get("name", "Unknown")
                            files = version_info.get("files", [])
                            if files:
                                download_url = files[0].get("downloadUrl")
                                filename = files[0].get("name")
                                detected_type = "Checkpoint" # It's a checkpoint
                                save_dir = self.prepare_download_path(self.subfolder_map.get(detected_type, "auto"), filename)
                                save_path = os.path.join(save_dir, filename)
                                log_func(f"Checkpoint will be saved to: {os.path.abspath(save_path)} (type: {detected_type})")
                                if os.path.exists(save_path) and not force_download:
                                    log_func(f"Checkpoint already exists: {filename}")
                                    self._saved_files.append(os.path.abspath(save_path))
                                else:
                                    log_func(f"Downloading checkpoint: {model_name} - {version_name}")
                                    headers = {"Authorization": f"Bearer {civitai_api_key}"} if civitai_api_key else {}
                                    download_file_with_status(download_url, save_path, log_func, headers, self)
                                    self._saved_files.append(os.path.abspath(save_path))
                            else:
                                log_func(f"❌ No files found for checkpoint version {version_id}")
                        except Exception as e:
                            log_func(f"❌ Error downloading checkpoint: {str(e)}")
                for i, lora in enumerate(loras):
                    version_id = lora.get("modelVersionId")
                    if version_id:
                        try:
                            log_func(f"Fetching LoRA version info for ID {version_id}...")
                            version_info = fetch_model_version_info(version_id, civitai_api_key)
                            model_name = version_info.get("model", {}).get("name", "Unknown")
                            version_name = version_info.get("name", "Unknown")
                            files = version_info.get("files", [])
                            if files:
                                download_url = files[0].get("downloadUrl")
                                filename = files[0].get("name")
                                detected_type = "LoRA" # It's a LoRA
                                save_dir = self.prepare_download_path(self.subfolder_map.get(detected_type, "loras"), filename)
                                save_path = os.path.join(save_dir, filename)
                                log_func(f"LoRA will be saved to: {os.path.abspath(save_path)} (type: {detected_type})")
                                if os.path.exists(save_path) and not force_download:
                                    log_func(f"LoRA already exists: {filename}")
                                    self._saved_files.append(os.path.abspath(save_path))
                                else:
                                    log_func(f"Downloading LoRA {i+1}/{len(loras)}: {model_name} - {version_name}")
                                    headers = {"Authorization": f"Bearer {civitai_api_key}"} if civitai_api_key else {}
                                    download_file_with_status(download_url, save_path, log_func, headers, self)
                                    self._saved_files.append(os.path.abspath(save_path))
                            else:
                                log_func(f"❌ No files found for LoRA version {version_id}")
                        except Exception as e:
                            log_func(f"❌ Error downloading LoRA {i+1}: {str(e)}")
                log_func("🎉 All assets from Civitai image processed!")
                if self._saved_files:
                    log_func(f"Summary of saved files:")
                    for f in self._saved_files:
                        log_func(f"  - {f}")
                else:
                    log_func("⚠️  No files were saved. This might be because they already exist. Use 'force_download' if needed.")
                self.set_progress(100)
                return ("\n".join(self._log_lines),)
            except Exception as e:
                error_msg = f"❌ Error processing Civitai image URL: {str(e)}"
                log_func(error_msg)
                return (error_msg,)
        # --- EXISTING: Handle direct asset URLs ---
        elif "civitai.com" in domain:
            result = self.download_from_civitai(asset_url, asset_type, base_path, force_download, civitai_api_key, log_func)
            log_func(result[0])
            return ("\n".join(self._log_lines),)
        elif "huggingface.co" in domain:
            result = self.download_from_huggingface(asset_url, asset_type, base_path, force_download, hf_token, log_func)
            log_func(result[0])
            return ("\n".join(self._log_lines),)
        elif "mega.nz" in domain:
            result = self.download_from_mega(asset_url, asset_type, base_path, force_download)
            log_func(result[0])
            return ("\n".join(self._log_lines),)
        else:
            return ("❌ Unsupported URL domain. Supported: civitai.com, huggingface.co, mega.nz",)

    def download_from_civitai(self, url, asset_type, base_path, force_download, api_key, log_func):
        try:
            model_id = self._extract_civitai_model_id(url)
            if not model_id:
                return ("❌ Could not extract model ID from Civitai URL",)
            api_url = f"https://civitai.com/api/v1/models/{model_id}"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            model_data = response.json()
            model_versions = model_data.get("modelVersions", [])
            if not model_versions:
                return ("❌ No versions found for this model",)
            latest_version = model_versions[0]
            files = latest_version.get("files", [])
            if not files:
                return ("❌ No files found for this model version",)
            file_info = files[0]
            download_url = file_info["downloadUrl"]
            filename = file_info["name"]
            # --- Asset type detection ---
            detected_type = asset_type
            if asset_type == "Auto":
                detected_type = self._detect_asset_type(model_data, file_info)
            save_dir = self.prepare_download_path(self.subfolder_map.get(detected_type, "checkpoints"), filename)
            save_path = os.path.join(save_dir, filename)
            if os.path.exists(save_path) and not force_download:
                return (f"✅ File already exists: {filename}",)
            
            download_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            success = download_file_with_status(download_url, save_path, log_func, download_headers, self)
            if success:
                return (f"✅ Successfully downloaded: {filename}",)
            else:
                return (f"❌ Failed to download: {filename}",)
        except Exception as e:
            return (f"❌ Error downloading from Civitai: {str(e)}",)

    def _extract_civitai_model_id(self, url):
        # Extract model ID from various Civitai URL formats
        patterns = [
            r"civitai.com/models/(\d+)",
            r"civitai.com/api/v1/models/(\d+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def download_from_huggingface(self, url, asset_type, base_path, force_download, hf_token, log_func):
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return ("❌ huggingface_hub is not installed. Please install it to use Hugging Face downloads.",)
        try:
            log_func("\n" + "="*60)
            log_func("🤗 HUGGING FACE ASSET DOWNLOAD")
            log_func("="*60)
            log_func(f"Parsing Hugging Face URL: {url}")

            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            repo_id = None
            file_path_in_repo = None
            revision = "main"

            # Handle blob/resolve URLs
            if len(path_parts) >= 4 and path_parts[2] in ['blob', 'resolve']:
                repo_id = f"{path_parts[0]}/{path_parts[1]}"
                revision = path_parts[3]
                file_path_in_repo = '/'.join(path_parts[4:])
            elif len(path_parts) >= 2:
                repo_id = f"{path_parts[0]}/{path_parts[1]}"
            else:
                log_func(f"❌ Invalid Hugging Face URL: {url}")
                return (f"❌ Invalid Hugging Face URL: {url}",)

            api = HfApi()
            try:
                info = api.model_info(repo_id, token=hf_token)
                log_func(f"Repo total downloads: {info.downloads}")
            except Exception as e:
                log_func(f"(Could not fetch repo download count: {e})")

            if not file_path_in_repo:
                log_func(f"Listing files in repo: {repo_id}")
                files = api.list_repo_files(repo_id=repo_id, revision=revision, token=hf_token)
                safetensor_files = [f for f in files if f.lower().endswith('.safetensors')]
                if not safetensor_files:
                    log_func(f"❌ No .safetensors files found in {repo_id}. Trying other common extensions.")
                    other_files = [f for f in files if f.lower().endswith(('.bin', '.pth', '.ckpt', '.gguf'))]
                    if not other_files:
                        return (f"❌ No suitable model files (.safetensors, .bin, .pth, .ckpt, .gguf) found in {repo_id}",)
                    safetensor_files = other_files
                log_func(f"Found {len(safetensor_files)} model files:")
                for f in safetensor_files:
                    log_func(f"  - {f}")
                def score(f):
                    s = 0
                    if '/' not in f: s += 10
                    if re.search(r'(unet|checkpoint|sdxl|model)', f, re.I): s += 5
                    if re.search(r'(vae|text_encoder|clip)', f, re.I): s -= 5
                    return s
                safetensor_files.sort(key=score, reverse=True)
                file_path_in_repo = safetensor_files[0]
                log_func(f"ℹ️ Auto-selected file: {file_path_in_repo}")

            filename = os.path.basename(file_path_in_repo)
            detected_type = asset_type
            if asset_type == "Auto":
                fname = filename.lower()
                if 'vae' in fname: detected_type = 'VAE'
                elif 'lora' in fname: detected_type = 'LoRA'
                elif 'controlnet' in fname or 'control' in fname: detected_type = 'ControlNet'
                elif 'clip' in fname: detected_type = 'CLIP'
                elif 'unet' in fname: detected_type = 'UNET'
                elif 'embedding' in fname or 'ti' in fname: detected_type = 'TextualInversion'
                elif 'gguf' in fname: detected_type = 'Checkpoint'
                else: detected_type = 'Checkpoint'

            save_dir = self.prepare_download_path(self.subfolder_map.get(detected_type, "auto"), "")
            save_path = os.path.join(save_dir, filename)
            log_func(f"Asset will be saved to: {os.path.abspath(save_path)} (type: {detected_type})")

            if os.path.exists(save_path) and not force_download:
                log_func(f"✅ File already exists: {filename}")
                return (f"✅ File already exists: {filename}",)

            # Build direct download URL
            download_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path_in_repo}"
            log_func(f"Downloading from Hugging Face: {repo_id}/{file_path_in_repo}")
            log_func(f"Direct download URL: {download_url}")

            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            success = download_file_with_status(download_url, save_path, log_func, headers, self)
            if success:
                return (f"✅ Successfully downloaded: {filename}",)
            else:
                return (f"❌ Failed to download: {filename}",)
        except Exception as e:
            log_func(f"❌ Error downloading from Hugging Face: {str(e)}")
            return (f"❌ Error downloading from Hugging Face: {str(e)}",)

    def _parse_hf_url(self, url):
        pattern = r"huggingface\.co/([^/]+/[^/]+)"
        match = re.search(pattern, url)
        if match:
            repo_id = match.group(1)
            file_path = url.split("/resolve/")[-1] if "/resolve/" in url else None
            return repo_id, file_path
        return None, None

    def _find_safetensors_file(self, files_data, token):
        for file_info in files_data:
            if file_info["type"] == "file" and file_info["name"].endswith(".safetensors"):
                return file_info
        return None

    def _detect_hf_asset_type(self, filename):
        filename_lower = filename.lower()
        if "lora" in filename_lower: return "LoRA"
        elif "vae" in filename_lower: return "VAE"
        elif "controlnet" in filename_lower or "control" in filename_lower: return "ControlNet"
        else: return "Checkpoint"

    def download_from_mega(self, url, asset_type, base_path, force_download):
        import shutil
        import os
        import re
        def log_func(msg):
            line = f"[MEGA Download] {msg}"
            print(line)
            self._log_lines.append(line)

        log_func("\n" + "="*60)
        log_func("🗄️  MEGA ASSET DOWNLOAD")
        log_func("="*60)
        log_func(f"Parsing MEGA URL: {url}")

        if not shutil.which('megadl'):
            log_func("❌ megadl (megatools) is not installed. Please install it to use MEGA downloads.")
            return ("❌ megadl (megatools) is not installed. Please install it to use MEGA downloads.",)

        filename = "mega_download.tmp"
        match = re.search(r'/([^/]+(\.safetensors|\.ckpt|\.bin|\.pth))', url)
        if match:
            filename = match.group(1)
        else:
            log_func(f"⚠️  Could not extract filename from URL, using a temporary name. This is risky.")

        detected_type = asset_type
        if asset_type == "Auto":
            log_func("⚠️  Asset type 'Auto' is not recommended for MEGA. Defaulting to 'Checkpoint'.")
            detected_type = self._detect_hf_asset_type(filename) # Re-use HF detection logic

        save_dir = self.prepare_download_path(self.subfolder_map.get(detected_type, "auto"), "")
        save_path = os.path.join(save_dir, filename)
        log_func(f"Asset will be saved to: {os.path.abspath(save_path)} (type: {detected_type})")

        if os.path.exists(save_path) and not force_download:
            log_func(f"✅ File already exists: {filename}")
            return (f"✅ File already exists: {filename}",)

        log_func(f"Downloading from MEGA to: {save_dir}")
        try:
            self._download_mega_cmd(url, save_dir, force_download)
            log_func(f"✅ Download completed: {filename}")
            log_func(f"File saved to: {os.path.abspath(save_path)}")
            return (f"✅ Successfully downloaded from MEGA: {filename}",)
        except Exception as e:
            log_func(f"❌ Download failed: {e}")
            return (f"❌ Download failed: {e}",)

    def _download_mega_cmd(self, url, destination_folder, force_download):
        cmd = ["megadl", url, "--path", destination_folder]
        if force_download:
            cmd.append("--force")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"megatools failed: {result.stderr}")