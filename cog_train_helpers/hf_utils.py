import os
import json
from pathlib import Path
from huggingface_hub import HfApi
from string import Template


def handle_hf_lora_filename(trigger_word: str, repo_id: str = None) -> str:
    """
    Create a standardized filename for the LoRA weights file.
    
    Args:
        trigger_word: The trigger word used during training
        repo_id: Optional Hugging Face repository ID
    """
    if trigger_word:
        # Use trigger word as basis for filename
        unique_part = trigger_word.lower().replace(" ", "-").replace("_", "-")[:32]
    elif repo_id:
        # Use repo name as fallback
        unique_part = repo_id.split("/")[-1].lower()
    else:
        # Default fallback
        unique_part = "lora"
    
    # Standard naming convention for WAN LoRAs
    return f"wan-14b-t2v-{unique_part}-lora.safetensors"


def handle_hf_readme(
    hf_repo_id: str,
    trigger_word: str,
    steps: int = 1000,
    learning_rate: float = 2e-5,
    lora_rank: int = 32,
    lora_filename: str = "wan-lora.safetensors",
    output_dir: str = None
) -> str:
    """
    Create a README file for the Hugging Face repository.
    
    Args:
        hf_repo_id: The Hugging Face repository ID
        trigger_word: The trigger word used during training
        steps: Number of training steps
        learning_rate: Learning rate used for training
        lora_rank: LoRA rank used for training
        lora_filename: Filename of the LoRA weights file
        output_dir: Directory where to save the README
    """
    # Use the standard output directory
    output_dir = output_dir or os.path.join("output", "wan_train_replicate")
    os.makedirs(output_dir, exist_ok=True)
    readme_path = os.path.join(output_dir, "README.md")
    
    # Look for template in standard locations
    template_path = os.path.join(output_dir, "hugging-face-readme-template.md")
    if not os.path.exists(template_path):
        template_path = "hugging-face-readme-template.md"
    
    # Format repo name for title
    repo_name = hf_repo_id.split("/")[-1].replace("-", " ").title() if "/" in hf_repo_id else hf_repo_id
    
    # If template exists, use it
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r') as file:
                template = Template(file.read())
            
            # Prepare template variables
            variables = {
                "repo_id": hf_repo_id,
                "title": repo_name,
                "trigger_word": trigger_word or "your_custom_subject",
                "trigger_section": f"\n## Trigger word\n\nYou should use `{trigger_word}` to trigger the video generation.\n" if trigger_word else "",
                "instance_prompt": f"instance_prompt: {trigger_word}" if trigger_word else "",
                "max_training_steps": steps,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "training_details": f"\n## Training details\n\n- Steps: {steps}\n- Learning rate: {learning_rate}\n- LoRA rank: {lora_rank}\n",
                "lora_filename": lora_filename.replace(".safetensors", ""),
                "model_type": "14B",
                "t2v_or_i2v": "T",
                "pipeline_tag": "text-to-video",
                "readable_finetuning_type": "Text-to-Video",
                "generation_code": "frames = base_model(\n    prompt=prompt,\n    negative_prompt=negative_prompt,\n    num_inference_steps=30,\n    guidance_scale=5.0,\n    width=832,\n    height=480,\n    fps=16,\n    num_frames=32,\n).frames[0]",
            }
            
            # Create the README file
            with open(readme_path, 'w') as file:
                file.write(template.substitute(variables))
                
            print(f"✅ Created README from template: {readme_path}")
        except Exception as e:
            print(f"⚠️ Template error, creating simple README: {str(e)}")
            create_simple_readme(readme_path, repo_name, trigger_word, steps, learning_rate, lora_rank, lora_filename)
    else:
        # No template found, create simple README
        print("⚠️ Template not found, creating simple README")
        create_simple_readme(readme_path, repo_name, trigger_word, steps, learning_rate, lora_rank, lora_filename)
    
    return readme_path


def create_simple_readme(readme_path, repo_name, trigger_word, steps, learning_rate, lora_rank, lora_filename):
    """Create a simple README without a template."""
    with open(readme_path, "w") as file:
        file.write(f"# {repo_name}\n\n")
        
        if trigger_word:
            file.write(f"## Trigger Word\n\nUse `{trigger_word}` to trigger this LoRA.\n\n")
            
        file.write("## Training Details\n\n")
        file.write(f"- Steps: {steps}\n")
        file.write(f"- Learning rate: {learning_rate}\n")
        file.write(f"- LoRA rank: {lora_rank}\n\n")
        
        file.write("## Usage\n\n")
        file.write(f"Use this LoRA with the WAN 14B model. The filename is `{lora_filename}`.\n")
    
    print(f"✅ Created simple README: {readme_path}")


def create_model_card_metadata(
    hf_repo_id: str, 
    model_type: str = "14b", 
    finetuning_type: str = "text2video", 
    trigger_word: str = None,
    output_dir: str = None
):
    """
    Create a model-card.json file with metadata for the Hugging Face model card.
    """
    # Use standard output directory
    output_dir = output_dir or os.path.join("output", "wan_train_replicate")
    
    # Create the metadata directory
    metadata_dir = os.path.join(output_dir, ".github")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Create basic metadata
    tags = ["wan", "video", "lora"]
    tags.append("text-to-video" if finetuning_type == "text2video" else "image-to-video")
    
    if trigger_word:
        tags.append(trigger_word.lower())
    
    metadata = {
        "language": "en",
        "license": "other",
        "model_name": hf_repo_id.split("/")[-1] if "/" in hf_repo_id else hf_repo_id,
        "base_model": f"Wan-AI/Wan2.1-{'I' if finetuning_type == 'image2video' else 'T'}2V-{model_type.upper()}-Diffusers",
        "tags": tags,
        "pipeline_tag": "image-to-video" if finetuning_type == "image2video" else "text-to-video",
        "library_name": "diffusers",
    }
    
    # Write metadata file
    metadata_path = os.path.join(metadata_dir, "model-card.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created model card metadata: {metadata_path}")
    return metadata_path 