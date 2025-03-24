import os
import json
from string import Template


def handle_hf_lora_filename(trigger_word: str, repo_id: str = None) -> str:
    """Create standardized filename for LoRA weights based on trigger word"""
    # Simplify by focusing on trigger_word only since it's always available
    clean_name = trigger_word.lower().replace(" ", "-").replace("_", "-")[:32]
    return f"wan-14b-t2v-{clean_name}-lora.safetensors"


def handle_hf_readme(
    hf_repo_id: str,
    trigger_word: str,
    steps: int, 
    learning_rate: float,
    lora_rank: int,
    lora_filename: str
) -> str:
    """Create README from template"""
    # Always use the standard output directory
    output_dir = os.path.join("output", "wan_train_replicate")
    os.makedirs(output_dir, exist_ok=True)
    readme_path = os.path.join(output_dir, "README.md")
    
    # Just look for the template in the root directory
    template_path = "hugging-face-readme-template.md"
    
    # Format repo name for title
    repo_name = hf_repo_id.split("/")[-1].replace("-", " ").title() if "/" in hf_repo_id else hf_repo_id
    
    # Use template if it exists, otherwise create simple README
    if os.path.exists(template_path):
        with open(template_path, 'r') as file:
            template = Template(file.read())
        
        # Prepare template variables
        variables = {
            "repo_id": hf_repo_id,
            "title": repo_name,
            "trigger_word": trigger_word,
            "trigger_section": f"\n## Trigger word\n\nYou should use `{trigger_word}` to trigger the video generation.\n",
            "instance_prompt": f"instance_prompt: {trigger_word}",
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
        
        # Create README
        with open(readme_path, 'w') as file:
            file.write(template.substitute(variables))
        
        # Copy template to output dir for reference
        import shutil
        shutil.copy(template_path, output_dir)
    else:
        # Simple fallback if template isn't found
        with open(readme_path, "w") as file:
            file.write(f"# {repo_name}\n\n")
            file.write(f"## Trigger Word\n\nUse `{trigger_word}` to trigger this LoRA.\n\n")
            file.write(f"## Training Details\n\n- Steps: {steps}\n- Learning rate: {learning_rate}\n- LoRA rank: {lora_rank}\n\n")
            file.write(f"## Usage\n\nUse this LoRA with the WAN 14B model. The filename is `{lora_filename}`.\n")
    
    return readme_path


def create_model_card_metadata(hf_repo_id: str, finetuning_type: str = "text2video"):
    """Create model card metadata file"""
    output_dir = os.path.join("output", "wan_train_replicate")
    metadata_dir = os.path.join(output_dir, ".github")
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata = {
        "language": "en",
        "license": "other",
        "model_name": hf_repo_id.split("/")[-1] if "/" in hf_repo_id else hf_repo_id,
        "base_model": f"Wan-AI/Wan2.1-{'I' if finetuning_type == 'image2video' else 'T'}2V-14B-Diffusers",
        "tags": ["wan", "video", "lora", "text-to-video" if finetuning_type == "text2video" else "image-to-video"],
        "pipeline_tag": "text-to-video" if finetuning_type == "text2video" else "image-to-video",
        "library_name": "diffusers",
    }
    
    metadata_path = os.path.join(metadata_dir, "model-card.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path 