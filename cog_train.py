import os
import json

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Import constants first so we can use them for environment setup
from cog_train_helpers.constants import MODEL_CACHE, QWEN_MODEL_CACHE, QWEN_MODEL_URL, INPUT_DIR, OUTPUT_DIR, JOB_NAME, BASE_URL

# Set model cache paths
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import shutil
import subprocess
import sys
import time
import toml
from pathlib import Path
from zipfile import ZipFile, is_zipfile
from cog import BaseModel, Input, Path as CogPath, Secret  # Added Secret import
from typing import Optional
import logging
import torch
import av
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import HfApi  # Added for HF uploads
from cog_train_helpers.gpu_utils import get_available_gpu_count, determine_optimal_gpu_count
from cog_train_helpers.model_utils import download_model, download_weights, setup_qwen_model
from cog_train_helpers.data_utils import extract_zip, autocaption_videos, add_trigger_word_to_captions
from cog_train_helpers.config_utils import create_dataset_toml, create_config_toml, handle_seed
from cog_train_helpers.training_utils import clean_up, run_training, archive_results
from cog_train_helpers.hf_utils import (
    handle_hf_lora_filename,
    handle_hf_readme,
    create_model_card_metadata
)

# Configure logging to suppress INFO messages
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Suppress common loggers
loggers_to_quiet = [
    "torch", "accelerate", "transformers", "__main__", "PIL",
    "safetensors", "xformers", "datasets", "tokenizers", 
    "diffusers", "filelock", "bitsandbytes", "qwen_vl_utils"
]

for logger_name in loggers_to_quiet:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Suppress third-party warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# We return a path to our trained adapter weights
class TrainingOutput(BaseModel):
    weights: CogPath


def train(
    input_video_zip: CogPath = Input(
        description="A zip file containing video and caption data for training. The zip should contain at least one video file (e.g., mp4, mov) and optionally caption files (.txt).",
        default=None,
    ),
    model_type: str = Input(
        description="Model size to use for training. '1.3b' is faster, '14b' gives higher quality but requires more VRAM.",
        default="14b",
        choices=["1.3b", "14b"],
    ),
    finetuning_type: str = Input(
        description="Choose training mode: 'text2video' learns to generate videos from text descriptions, 'image2video' learns to extend the first frame of a video into motion (requires 14B model).",
        default="text2video",
        choices=["text2video", "image2video"],
    ),
    video_clip_mode: str = Input(
        description="How to use your video during training: 'single_beginning' (focus on the opening scene), 'single_middle' (focus on the middle portion, ignoring intro/outro), or 'multiple_overlapping' (try to learn from the entire video by using multiple segments). For most cases, 'single_middle' gives the best results.",
        default="single_beginning",
        choices=["single_beginning", "single_middle", "multiple_overlapping"],
    ),
    trigger_word: str = Input(
        description="The trigger word to be associated with all videos during training. This word will help activate the LoRA when used in prompts.",
        default="TOK",
    ),
    autocaption: bool = Input(
        description="Automatically caption videos using QWEN-VL that don't have matching caption files.",
        default=True,
    ),
    autocaption_prefix: str = Input(
        description="Optional: Text you want to appear at the beginning of all your generated captions; for example, 'a video of TOK, '. You can include your trigger word in the prefix.",
        default="",
    ),
    autocaption_suffix: str = Input(
        description="Optional: Text you want to appear at the end of all your generated captions; for example, ' in the style of TOK'. You can include your trigger word in suffixes.",
        default="",
    ),
    max_training_steps: int = Input(
        description=(
            "Total number of training steps, including warmup. For example, if max_training_steps=1000 and "
            "warmup_steps_budget=100, then you have 1000 steps total, with the first 100 for warmup."
        ),
        default=1000,
        ge=10,
        le=20000,
    ),
    learning_rate: float = Input(
        description="Learning rate for training. Higher values may lead to faster convergence but potential instability.",
        default=2e-5,
        ge=1e-5,
        le=1e-2,
    ),
    lora_rank: int = Input(
        description="LoRA rank for training. Higher ranks can capture more complex features but require more training time.",
        default=32,
        ge=16,
        le=256,
    ),
    warmup_steps_budget: int = Input(
        description=(
            "If not provided or set to -1, defaults to 10% of max_training_steps. These steps ramp from a lower LR to the "
            "configured LR. They are included within max_training_steps, not added on top."
        ),
        default=-1,
        ge=-1,
        le=2000,
    ),
    weight_decay: float = Input(
        description="Weight decay for regularization. Controls overfitting - lower values allow more detailed memorization.",
        default=0.0001,
        ge=0.0,
        le=0.1,
    ),
    seed: int = Input(
        description="Random seed for training reproducibility. Use -1 for a random seed.",
        default=-1,
    ),
    hf_repo_id: str = Input(
        description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, username/wan-lora. If the given repo does not exist, a new public repo will be created.",
        default=None,
    ),
    hf_token: Secret = Input(
        description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
        default=None,
    ),
) -> TrainingOutput:
    """
    Train a WAN model adapter on your video using LoRA fine-tuning.
    """

    # If warmup_steps_budget not provided or -1, default to 10% of max_training_steps
    if warmup_steps_budget is None or warmup_steps_budget == -1:
        warmup_steps_budget = int(0.1 * max_training_steps)
    
    # Validate model type with finetuning type
    if finetuning_type == "image2video" and model_type != "14b":
        raise ValueError("Image-to-video finetuning requires the 14B model. Please select model_type='14b'.")
    
    # Auto-detect GPU count
    available_gpus = get_available_gpu_count()
    num_gpus = determine_optimal_gpu_count(model_type, available_gpus)
    print(f"\n=== 🖥️ GPU Auto-detection ===")
    print(f"  • Available GPUs: {available_gpus}")
    print(f"  • Using: {num_gpus} GPU(s)")
    print("=====================================\n")

    print("\n=== 🎥 WAN Video LoRA Training ===")
    print("📊 Configuration:")
    print(f"  • Input: {input_video_zip}")
    print(f"  • Model: {model_type}")
    print(f"  • Finetuning type: {finetuning_type}")
    print(f"  • GPUs: {num_gpus}")
    print(f"  • Training:")
    print(f"    - Max Steps: {max_training_steps}")
    print(f"    - Warmup Steps: {warmup_steps_budget}")
    print(f"    - LoRA Rank: {lora_rank}")
    print(f"    - Learning Rate: {learning_rate}")
    print(f"    - Weight Decay: {weight_decay}")
    print(f"    - Video Clip Mode: {video_clip_mode}")
    if autocaption:
        print(f"  • Auto-captioning:")
        print(f"    - Enabled: {autocaption}")
        print(f"    - Trigger Word: {trigger_word}")
        if autocaption_prefix:
            print(f"    - Prefix: {autocaption_prefix}")
        if autocaption_suffix:
            print(f"    - Suffix: {autocaption_suffix}")
    else:
        print(f"  • Auto-captioning: Disabled")
        print(f"  • Trigger Word: {trigger_word}")
    if hf_repo_id:
        print(f"  • Hugging Face Upload:")
        print(f"    - Repository: {hf_repo_id}")
    print("=====================================\n")

    if not input_video_zip:
        raise ValueError("You must provide a zip file containing training data.")

    if not is_zipfile(input_video_zip):
        raise ValueError("The provided input must be a zip file.")

    # Setup directories and seed
    clean_up()
    seed = handle_seed(seed)
    
    # Download the model with specified type
    download_model(model_type, finetuning_type)
    
    # Extract zip and set up data directory
    extract_zip(
        input_video_zip, 
        INPUT_DIR, 
        autocaption=autocaption,
        trigger_word=trigger_word,
        autocaption_prefix=autocaption_prefix,
        autocaption_suffix=autocaption_suffix
    )
    
    # Add trigger word to captions if not already done in autocaptioning
    if not autocaption:
        add_trigger_word_to_captions(trigger_word)
    
    # Create configuration files
    create_dataset_toml(video_clip_mode)
    create_config_toml(
        model_type=model_type,
        finetuning_type=finetuning_type,
        video_clip_mode=video_clip_mode,
        training_steps=max_training_steps,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        warmup_steps=warmup_steps_budget,
        weight_decay=weight_decay,
        seed=seed,
        num_gpus=num_gpus
    )
    
    # Run training - pass max_training_steps to force a hard stop
    run_training(max_training_steps, num_gpus)
    
    # Archive results
    output_path = archive_results()
    
    # Upload to Hugging Face if requested
    if hf_repo_id and hf_token:
        job_dir = os.path.join("output", "wan_train_replicate")
        
        if os.path.exists(job_dir):
            # Find and rename safetensors file
            safetensors_files = [f for f in os.listdir(job_dir) if f.endswith('.safetensors')]
            if safetensors_files:
                old_lora_file = safetensors_files[0]
                new_lora_file = handle_hf_lora_filename(trigger_word)
                os.rename(
                    os.path.join(job_dir, old_lora_file),
                    os.path.join(job_dir, new_lora_file)
                )
                
                # Create README and metadata
                handle_hf_readme(
                    hf_repo_id=hf_repo_id,
                    trigger_word=trigger_word,
                    steps=max_training_steps,
                    learning_rate=learning_rate,
                    lora_rank=lora_rank,
                    lora_filename=new_lora_file
                )
                
                create_model_card_metadata(hf_repo_id, finetuning_type)
                
                # Upload to HF
                print(f"Uploading to Hugging Face: {hf_repo_id}")
                api = HfApi()
                repo_url = api.create_repo(
                    hf_repo_id,
                    private=False,
                    exist_ok=True,
                    token=hf_token.get_secret_value()
                )
                
                # Upload the folder
                api.upload_folder(
                    repo_id=hf_repo_id,
                    folder_path=job_dir,
                    repo_type="model",
                    token=hf_token.get_secret_value()
                )
                
                print(f"🎉 Model uploaded to Hugging Face: {repo_url}")
    
    print("\n=== 🎉 Training Complete! ===")
    print(f"  • Trained model saved to: {output_path}")
    print(f"  • You can now use your WAN LoRA with trigger word: '{trigger_word}'")
    if hf_repo_id and hf_token and 'repo_url' in locals() and repo_url:
        print(f"  • Your model has been uploaded to Hugging Face: {repo_url}")
    elif hf_repo_id and hf_token:
        print(f"  • Attempted to upload to Hugging Face: {hf_repo_id}, but there was an error")
    print("=====================================")
    
    return TrainingOutput(weights=CogPath(output_path))
