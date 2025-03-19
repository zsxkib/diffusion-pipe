import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi


def handle_hf_lora_filename(trigger_word: str, model_type: str, finetuning_type: str, repo_id: str = None) -> str:
    """
    Create a standardized filename for the LoRA weights file to be uploaded to HF.
    
    Args:
        trigger_word: The trigger word used during training
        model_type: Model size (1.3b or 14b)
        finetuning_type: Text2video or image2video
        repo_id: Optional HF repo ID to use for naming
        
    Returns:
        Standardized filename for the LoRA weights
    """
    if trigger_word:
        unique_part = trigger_word.lower().replace(" ", "-").replace("_", "-")[:32]
    elif repo_id:
        unique_part = repo_id.split("/")[-1].lower()
        patterns = [
            "wan", "14b", "1.3b", "1_3b", "text2video", "image2video", "t2v", "i2v"
        ]
        for pattern in patterns:
            for sep in ["-", "_"]:
                unique_part = unique_part.replace(f"{pattern}{sep}".lower(), "")
    else:
        unique_part = "wan-lora"
    
    ft_type_str = "i2v" if finetuning_type == "image2video" else "t2v"
    
    return f"wan-{model_type}-{ft_type_str}-{unique_part}-lora.safetensors"


def handle_hf_readme(hf_repo_id: str, trigger_word: str, model_type: str, finetuning_type: str, 
                   max_training_steps: int, learning_rate: float, lora_rank: int, 
                   lora_filename: str, output_dir: str) -> str:
    """
    Create a README file for the Hugging Face repository.
    
    Args:
        hf_repo_id: The Hugging Face repository ID
        trigger_word: The trigger word used during training
        model_type: Model size (1.3b or 14b)
        finetuning_type: Text2video or image2video
        max_training_steps: Number of training steps
        learning_rate: Learning rate used for training
        lora_rank: LoRA rank used for training
        lora_filename: Filename of the LoRA weights file
        output_dir: Directory where to save the README
        
    Returns:
        Path to the created README file
    """
    # Create README content
    repo_name = hf_repo_id.split("/")[-1].replace("-", " ").title() if "/" in hf_repo_id else hf_repo_id
    ft_type_readable = "Image-to-Video" if finetuning_type == "image2video" else "Text-to-Video"
    base_model_type = model_type.upper()
    
    # Get base model path for usage example
    base_model = f"Wan-AI/Wan2.1-{ft_type_readable[:1]}2V-{base_model_type}-Diffusers"
    
    readme_content = f"""# {repo_name}

This repository contains a fine-tuned LoRA for the WAN {base_model_type} {ft_type_readable} model.

## Model details

- **Base model**: [WAN {base_model_type} {ft_type_readable}](https://huggingface.co/{base_model})
- **Training steps**: {max_training_steps}
- **Learning rate**: {learning_rate}
- **LoRA rank**: {lora_rank}
"""

    if trigger_word:
        readme_content += f"""
## Trigger word

To use this LoRA effectively, include the trigger word `{trigger_word}` in your prompts.
"""

    # Build usage example based on model type and finetuning type
    adapter_path = f"{hf_repo_id}" if hf_repo_id else "path/to/adapter"
    
    if finetuning_type == "text2video":
        prompt = f"a video of {trigger_word if trigger_word else 'your subject'}"
        usage_example = f"""
## Usage example

```python
import torch
from diffusers import WanVidAdapter, WanVid
from diffusers.utils import export_to_video

# Load base model
base_model = WanVid.from_pretrained("{base_model}", torch_dtype=torch.float16)

# Load and apply LoRA adapter
adapter = WanVidAdapter.from_pretrained("{adapter_path}")
base_model.load_adapter(adapter)

# Generate video
prompt = "{prompt}"
negative_prompt = "blurry, low quality, low resolution"

# Generate video frames
frames = base_model(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=5.0,
    width=832,
    height=480,
    fps=16,
    num_frames=32,
).frames[0]

# Save as video
video_path = "output_video.mp4"
export_to_video(frames, video_path, fps=16)
print(f"Video saved to: {{video_path}}")
```
"""
    else:  # image2video
        usage_example = f"""
## Usage example

```python
import torch
from diffusers import WanVidAdapter, WanVid
from diffusers.utils import export_to_video
from PIL import Image

# Load base model
base_model = WanVid.from_pretrained("{base_model}", torch_dtype=torch.float16)

# Load and apply LoRA adapter
adapter = WanVidAdapter.from_pretrained("{adapter_path}")
base_model.load_adapter(adapter)

# Load input image
image = Image.open("path/to/your/image.jpg").convert("RGB")

# Optional: Add text prompt to guide the generation
prompt = "a video of {trigger_word if trigger_word else 'your subject'}"
negative_prompt = "blurry, low quality, low resolution"

# Generate video from image
frames = base_model.image_to_video(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=5.0,
    fps=16,
    num_frames=32,
).frames[0]

# Save as video
video_path = "output_video.mp4"
export_to_video(frames, video_path, fps=16)
print(f"Video saved to: {{video_path}}")
```
"""

    readme_content += usage_example
    
    # Add license and citation information
    readme_content += """
## License

This model is shared under the terms of the license of the original WAN model.

## Citation

If you use this model in your research, please cite the WAN paper:
```
@article{stypulkowski2024wan,
  title={WAN: A Wonder of Anime},
  author={Stypulkowski, Chen and Sharma, Naveen and Lai, Wenhan and Singh, Satpreet and Kommineni, Vivek and Freer, Chris and Weber, Gunnar and Tang, Alex and Cao, Zach and Wang, Chung-Yi and Chechik, Gunnar},
  journal={arXiv preprint arXiv:2403.18257},
  year={2024}
}
```
"""

    # Create the README file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    return readme_path


def upload_to_huggingface(hf_repo_id: str, hf_token: str, output_dir: str, lora_filename: str) -> str:
    """
    Upload the trained model to Hugging Face.
    
    Args:
        hf_repo_id: The Hugging Face repository ID
        hf_token: Hugging Face API token
        output_dir: Directory containing the files to upload
        lora_filename: Filename of the LoRA weights file
        
    Returns:
        URL of the created repository
    """
    try:
        print(f"\n=== 🤗 Uploading to Hugging Face ===")
        print(f"  • Repository: {hf_repo_id}")
        print(f"  • LoRA file: {lora_filename}")
        
        api = HfApi()
        
        # Create or get the repository
        repo_url = api.create_repo(
            hf_repo_id,
            private=False,
            exist_ok=True,
            token=hf_token,
        )
        
        print(f"  • Repository URL: {repo_url}")
        
        # Upload the folder contents
        api.upload_folder(
            repo_id=hf_repo_id,
            folder_path=output_dir,
            repo_type="model",
            use_auth_token=hf_token,
        )
        
        print(f"✅ Upload complete!")
        print(f"  • Your model is available at: {repo_url}")
        print("=====================================\n")
        
        return repo_url
    except Exception as e:
        print(f"⚠️ Error uploading to Hugging Face: {str(e)}")
        print("=====================================\n")
        return None


def create_model_card_metadata(
    hf_repo_id: str, 
    model_type: str, 
    finetuning_type: str, 
    trigger_word: str,
    output_dir: str
) -> str:
    """
    Create a model-card.json file with metadata for the Hugging Face model card.
    
    Args:
        hf_repo_id: Hugging Face repository ID
        model_type: Model size (1.3b or 14b)
        finetuning_type: Text2video or image2video
        trigger_word: Optional trigger word used during training
        output_dir: Directory where to save the metadata
        
    Returns:
        Path to the created metadata file
    """
    # Create the metadata directory
    metadata_dir = os.path.join(output_dir, ".github")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Determine model information
    ft_type_readable = "Image-to-Video" if finetuning_type == "image2video" else "Text-to-Video"
    base_model = f"Wan-AI/Wan2.1-{ft_type_readable[:1]}2V-{model_type.upper()}-Diffusers"
    
    # Create the metadata content
    tags = [
        "wan", 
        "video", 
        "text-to-video" if finetuning_type == "text2video" else "image-to-video",
        f"wan-{model_type}",
        "lora"
    ]
    
    if trigger_word:
        tags.append(trigger_word.lower())
    
    metadata = {
        "language": "en",
        "license": "other",
        "model_name": hf_repo_id.split("/")[-1] if "/" in hf_repo_id else hf_repo_id,
        "base_model": base_model,
        "tags": tags,
        "pipeline_tag": "text-to-video" if finetuning_type == "text2video" else "image-to-video",
        "library_name": "diffusers",
    }
    
    # Write metadata file
    metadata_path = os.path.join(metadata_dir, "model-card.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path 