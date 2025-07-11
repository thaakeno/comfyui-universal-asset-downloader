# ComfyUI Universal Asset Downloader

A robust, user-friendly, and intelligent custom node for ComfyUI. This "Universal Asset Downloader" streamlines the process of downloading and organizing AI model assets from Civitai, Hugging Face, and MEGA directly within the ComfyUI interface.

![image](https://github.com/user-attachments/assets/04bbacd3-23c3-4f1b-94fb-0a71ad40eef9)


## Features
- **Multi-Platform Support:** Download assets from Civitai, Hugging Face, and MEGA.
- **Intelligent URL Detection:** Automatically identify the source platform from a given URL.
- **Smart Asset Type-Detection:** Automatically determine the correct asset type (e.g., Checkpoint, LoRA, VAE) and place it in the corresponding ComfyUI directory.
- **API Integration:** Utilizes official APIs for Civitai and Hugging Face to fetch metadata and ensure reliable downloads.
- **Progress Tracking:** Displays a visual progress bar for downloads.
- **Error Handling:** Provides clear, informative error messages.

## Installation

1.  **Clone the Repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```bash
    git clone https://github.com/thaakeno/comfyui-universal-asset-downloader.git
    ```
    (Replace `your-username` with your actual GitHub username)

2.  **Install Dependencies:**
    Navigate into the newly created folder and install the required Python packages:
    ```bash
    cd comfyui-universal-asset-downloader
    pip install -r requirements.txt
    ```
    For MEGA support, it is also recommended to install `mega-cmd` from the official website for your operating system. The node will fall back to `mega.py` if it's not found.

3.  **Restart ComfyUI:**
    Restart your ComfyUI server. The new node will be loaded automatically.

## Usage

1.  **Add the Node:** In the ComfyUI editor, right-click and select `Add Node` -> `utilities/downloaders` -> `🌐 Universal Asset Downloader`.
2.  **Paste URL:** Copy the URL of the asset you want to download from Civitai, Hugging Face, or MEGA and paste it into the `asset_url` field.
3.  **Select Asset Type:**
    *   Leave it as `"Auto"` for the node to intelligently detect the correct asset type.
    *   Manually select the type if you want to override the detection.
4.  **Provide API Keys (Optional):** For private repositories or to avoid rate limits, provide your Civitai and/or Hugging Face API keys.
5.  **Execute:** Queue the prompt. The node will download the file to the correct subfolder within your ComfyUI models directory and output the final file path.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the GNU AGPLv3. See the [LICENSE](LICENSE) file for details.
