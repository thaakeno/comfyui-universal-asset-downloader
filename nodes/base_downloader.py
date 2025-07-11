from server import PromptServer
import os

class BaseDownloader:
    # This makes the node runnable on its own and allows it to be queued
    OUTPUT_NODE = True
    CATEGORY = "utilities/downloaders"

    def __init__(self):
        self.node_id = None

    def set_progress(self, percentage):
        # Sends progress updates to the ComfyUI frontend
        if self.node_id:
            PromptServer.instance.send_sync("progress", {
                "node": self.node_id,
                "value": percentage,
                "max": 100
            })

    def prepare_download_path(self, local_path, filename):
        # This needs to be adapted to ComfyUI's models directory structure
        # Assuming base_path is the ComfyUI root or a custom model dir
        # For now, we'll keep it simple, but this might need refinement
        # depending on where ComfyUI expects specific model types.
        full_path = os.path.join(local_path, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path
    
    def handle_download(self, download_func, save_path, filename, overwrite=False, **kwargs):
        file_path = os.path.join(save_path, filename)
        
        if os.path.exists(file_path) and not overwrite:
            self.set_progress(100) # Mark as complete if already exists
            return (f"✅ File already exists: {file_path}", file_path)
        
        try:
            self.set_progress(0) # Start progress
            kwargs['save_path'] = save_path
            kwargs['progress_callback'] = self # Pass self as progress callback
            
            download_func(**kwargs)
            self.set_progress(100) # Mark as complete
            return (f"✅ Downloaded: {file_path}", file_path)
        except Exception as e:
            self.set_progress(0) # Reset progress on failure
            print(f"❌ Download failed: {str(e)}")
            return (f"❌ Download failed: {str(e)}", "")