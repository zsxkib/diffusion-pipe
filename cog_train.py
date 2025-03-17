import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Set model cache paths
MODEL_CACHE = "./model_cache"
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
from cog import BaseModel, Input, Path as CogPath  # Removed Secret import
from typing import Optional
import logging
import torch
import av
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Constants for Qwen2-VL model
QWEN_MODEL_CACHE = "./qwen_checkpoints"
QWEN_MODEL_URL = "https://weights.replicate.delivery/default/qwen/Qwen2-VL-7B-Instruct/model.tar"

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

# Constants
INPUT_DIR = "data"
OUTPUT_DIR = "output_wan"
JOB_NAME = "wan_train_replicate"

BASE_URL = "https://weights.replicate.delivery/default/wan2.1/model_cache/"

def get_available_gpu_count():
    """Detect the number of available GPUs."""
    try:
        return torch.cuda.device_count()
    except:
        return 0


def determine_optimal_gpu_count(model_type, available_gpus):
    """Determine the optimal number of GPUs to use based on model size and availability."""
    # For 14B models, use multiple GPUs if available (up to 4)
    if model_type.lower() == "14b":
        # Use all available GPUs up to 4 for 14B models
        return min(available_gpus, 4)
    else:
        # For 1.3B models, one GPU is usually sufficient
        return 1


def download_weights(url: str, dest: str) -> None:
    """Download weights from URL to destination path."""
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    
    if ".tar" in dest:
        dest = os.path.dirname(dest)
        
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    
    print(f"[~] Running command: {' '.join(command)}")
    subprocess.check_call(command, close_fds=False)
    
    print("[!] Download took:", time.time() - start, "seconds")


def download_model(model_type: str = "1.3b", finetuning_type: str = "text2video") -> None:
    """Download model weights based on specified model type and finetuning type."""
    print("\n=== 📥 Downloading WAN Model ===")
    print(f"Model type: {model_type}")
    print(f"Finetuning type: {finetuning_type}")
    
    if finetuning_type == "image2video":
        if model_type.lower() != "14b":
            raise ValueError("Image to video finetuning requires the 14B model.")
        # Download only the I2V model for image2video finetuning
        model_files = ["Wan2.1-I2V-14B-480P.tar"]
        print("Selected I2V-14B-480P model for image-to-video training")
    else:  # text2video
        if model_type.lower() == "1.3b":
            model_files = ["Wan2.1-T2V-1.3B.tar"]
            print("Selected 1.3B model for text-to-video training")
        elif model_type.lower() == "14b":
            model_files = ["Wan2.1-T2V-14B.tar"]
            print("Selected 14B model for text-to-video training")
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose '1.3b' or '14b'.")
    
    if not os.path.exists(MODEL_CACHE):
        os.makedirs(MODEL_CACHE)
    
    for model_file in model_files:
        url = BASE_URL + model_file
        filename = url.split("/")[-1]
        dest_path = os.path.join(MODEL_CACHE, filename)
        extracted_dir = dest_path.replace(".tar", "")
        
        # Define required files for each model type to verify complete download
        required_files = ['config.json']
        
        # Check if the extracted directory exists and has the required files
        is_complete = os.path.exists(extracted_dir) and all(
            os.path.exists(os.path.join(extracted_dir, req_file)) 
            for req_file in required_files
        )
        
        if not is_complete:
            # If directory exists but is incomplete, remove it before downloading
            if os.path.exists(extracted_dir):
                print(f"Model {model_file} appears incomplete, cleaning up and re-downloading...")
                shutil.rmtree(extracted_dir)
            # Also remove the tar file if it exists but extraction was incomplete
            if os.path.exists(dest_path):
                print(f"Removing potentially corrupted tar file: {dest_path}")
                os.remove(dest_path)
                
            print(f"Downloading {model_file}...")
            download_weights(url, dest_path)
            
            # Verify the extraction was successful
            if not all(os.path.exists(os.path.join(extracted_dir, req_file)) for req_file in required_files):
                print(f"⚠️ Warning: Model {model_file} may still be incomplete after download.")
                print(f"Missing files in {extracted_dir}:")
                for req_file in required_files:
                    if not os.path.exists(os.path.join(extracted_dir, req_file)):
                        print(f"  - {req_file}")
        else:
            print(f"Model {model_file} already exists and appears complete, skipping download")
    
    # Final verification for the main model that will be used for training
    if finetuning_type == "image2video":
        main_model_dir = os.path.join(MODEL_CACHE, "Wan2.1-I2V-14B-480P")
    else:
        main_model_dir = os.path.join(MODEL_CACHE, f"Wan2.1-T2V-{model_type.upper()}")
    
    if not os.path.exists(os.path.join(main_model_dir, 'config.json')):
        raise ValueError(f"Critical error: The main model at {main_model_dir} is still missing config.json after download attempts.")
    
    print(f"✅ Model download check completed for {model_type} model(s)")
    print("=====================================")


def train(
    input_video_zip: CogPath = Input(
        description="A zip file containing video and caption data for training. The zip should contain at least one video file (e.g., mp4, mov) and optionally caption files (.txt).",
        default=None,
    ),
    model_type: str = Input(
        description="Model size to use for training. '1.3b' is faster, '14b' gives higher quality but requires more VRAM.",
        default="1.3b",
        choices=["1.3b", "14b"],
    ),
    finetuning_type: str = Input(
        description="Choose training mode: 'text2video' learns to generate videos from text descriptions, 'image2video' learns to extend the first frame of a video into motion (requires 14B model).",
        default="text2video",
        choices=["text2video", "image2video"],
    ),
    video_clip_mode: str = Input(
        description="How to use your video during training: 'single_beginning' (focus on the opening scene), 'single_middle' (focus on the middle portion, ignoring intro/outro), or 'multiple_overlapping' (try to learn from the entire video by using multiple segments). For most cases, 'single_middle' gives the best results.",
        default="single_middle",
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
        default=5e-4,
        ge=1e-5,
        le=1e-2,
    ),
    lora_rank: int = Input(
        description="LoRA rank for training. Higher ranks can capture more complex features but require more training time.",
        default=64,
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
    
    print("\n=== 🎉 Training Complete! ===")
    print(f"  • Trained model saved to: {output_path}")
    print(f"  • You can now use your WAN LoRA with trigger word: '{trigger_word}'")
    print("=====================================\n")
    
    # Return the path to the trained weights
    return TrainingOutput(weights=CogPath(output_path))


def handle_seed(seed: int) -> int:
    """Set up random seed, generating one if seed is -1."""
    print("\n=== 🎲 Setting Random Seed ===")
    if seed == -1:
        seed = int.from_bytes(os.urandom(2), "big")
        print(f"Generated random seed: {seed}")
    else:
        print(f"Using provided seed: {seed}")
    print("=====================================")
    return seed


def clean_up() -> None:
    """Clean up existing directories before training."""
    print("\n=== 🧹 Cleaning Up Previous Data ===")
    # Clean directories
    for dir in [INPUT_DIR, OUTPUT_DIR, QWEN_MODEL_CACHE]:
        if os.path.exists(dir):
            print(f"Removing existing directory: {dir}")
            shutil.rmtree(dir)
    print("✅ Workspace is clean")
    print("=====================================")


def extract_zip(
    zip_path: CogPath, 
    input_dir: str, 
    autocaption: bool = False,
    trigger_word: Optional[str] = None,
    autocaption_prefix: Optional[str] = None,
    autocaption_suffix: Optional[str] = None
) -> None:
    """Extract training data from zip file, handling various input structures robustly."""
    print("\n=== 📦 Extracting Zip File ===")
    
    if not is_zipfile(zip_path):
        raise ValueError("The provided input must be a zip file.")
    
    # Create target directory
    target_dir = os.path.join(input_dir, JOB_NAME)  # Use JOB_NAME constant for consistency
    os.makedirs(target_dir, exist_ok=True)
    
    # Track file types and counts
    video_files = set()
    caption_files = set()
    video_count = 0
    text_count = 0
    
    # Extract the zip file to a temporary location first
    temp_extract_dir = os.path.join(input_dir, "temp_extract")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir)
    
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    
    # Find all video and caption files in the extracted content
    for root, _, files in os.walk(temp_extract_dir):
        for file in files:
            # Skip mac-specific hidden files
            if file.startswith("._") or "__MACOSX" in root:
                continue
                
            file_path = os.path.join(root, file)
            base_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1].lower()
            
            # Move to the target directory with a flat structure
            dest_path = os.path.join(target_dir, file)
            
            # If file with same name exists, add a unique suffix
            if os.path.exists(dest_path):
                filename, ext = os.path.splitext(file)
                dest_path = os.path.join(target_dir, f"{filename}_{int(time.time())}{ext}")
            
            shutil.copy2(file_path, dest_path)
            
            # Track by file type
            if ext.lower() in ('.mp4', '.mov', '.avi', '.mkv'):
                video_files.add(base_name)
                video_count += 1
            elif ext.lower() == '.txt':
                caption_files.add(base_name)
                text_count += 1
    
    # Clean up temporary directory
    shutil.rmtree(temp_extract_dir)
    
    # Handle auto-captioning if enabled
    if autocaption and video_count > 0:
        caption_files = autocaption_videos(
            target_dir,
            video_files,
            caption_files,
            trigger_word,
            autocaption_prefix,
            autocaption_suffix
        )
    
    # Validate content
    if video_count == 0:
        raise ValueError("No video files found in the zip file. Please include at least one video file.")
    
    # Report on caption status
    videos_without_captions = video_files - caption_files
    captions_without_videos = caption_files - video_files
    matched_pairs = video_files & caption_files
    
    print(f"✅ Extracted {video_count} video files and {text_count} caption files")
    print(f"📊 Content status:")
    print(f"  • Valid Pairs: {len(matched_pairs)} video-caption pairs")
    if videos_without_captions:
        print(f"  • ⚠️ Missing Captions: {len(videos_without_captions)} videos")
        for v in sorted(videos_without_captions):
            print(f"    - {v}")
    if captions_without_videos:
        print(f"  • ⚠️ Orphaned Captions: {len(captions_without_videos)} files")
    
    # Create cache directory structure
    os.makedirs(os.path.join(target_dir, "cache", "wan", "cache_512x512x16"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "cache", "wan", "metadata"), exist_ok=True)
    
    print(f"✅ Setup complete at {target_dir}")
    print("=====================================")


def setup_qwen_model():
    """Download and setup Qwen2-VL model for auto-captioning"""
    print("\n=== 🧠 Setting Up QWEN-VL Model ===")
    if not os.path.exists(QWEN_MODEL_CACHE):
        print(f"Downloading Qwen2-VL model to {QWEN_MODEL_CACHE}")
        start = time.time()
        download_weights(QWEN_MODEL_URL, QWEN_MODEL_CACHE)
        print(f"Download took: {time.time() - start:.2f}s")
    else:
        print(f"Using existing Qwen2-VL model from {QWEN_MODEL_CACHE}")

    print("Loading QWEN model into GPU memory...")
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_CACHE)
        print("✅ QWEN-VL model loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading QWEN model with flash attention: {e}")
        print("Trying again without flash attention...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_CACHE,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_CACHE)
        print("✅ QWEN-VL model loaded successfully without flash attention")
    
    print("=====================================")
    return model, processor


def autocaption_videos(
    videos_path: str,
    video_files: set,
    caption_files: set,
    trigger_word: Optional[str] = None,
    autocaption_prefix: Optional[str] = None,
    autocaption_suffix: Optional[str] = None,
) -> set:
    """Generate captions for videos that don't have matching .txt files."""
    videos_without_captions = video_files - caption_files
    if not videos_without_captions:
        print("\n=== ✅ All videos already have captions ===")
        return caption_files

    print("\n=== 🤖 Auto-captioning Videos ===")
    print(f"Found {len(videos_without_captions)} videos without captions")
    
    model, processor = setup_qwen_model()
    
    new_caption_files = caption_files.copy()
    for i, vid_name in enumerate(videos_without_captions, 1):
        # Try to find a video file with this base name
        video_found = False
        for ext in ['.mp4', '.mov', '.avi', '.mkv']:
            video_path = os.path.join(videos_path, vid_name + ext)
            if os.path.exists(video_path):
                video_found = True
                print(f"\n[{i}/{len(videos_without_captions)}] 🎥 Processing: {vid_name}{ext}")

                # Use absolute path
                abs_path = os.path.abspath(video_path)

                # Build caption components
                prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
                suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
                trigger = f"{trigger_word} " if trigger_word else ""

                # First try to generate detailed prompt with video context
                try:
                    # Prepare messages format with customized prompt
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": abs_path,
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this video clip in detail, focusing on the key visual elements, actions, and overall scene.",
                                },
                            ],
                        }
                    ]

                    # Process input
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    # Check if we have video frames
                    if not video_inputs or len(video_inputs) == 0:
                        print("⚠️ No valid video frames extracted. Using fallback caption.")
                        raise ValueError("No video frames extracted")
                    
                    inputs = processor(
                        text=[text],
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")

                    print("Generating caption...")
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    caption = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    # Combine prefix, trigger, caption, and suffix
                    final_caption = f"{prefix}{trigger}{caption.strip()}{suffix}"
                    print("\n📝 Generated Caption:")
                    print("--------------------")
                    print(f"{final_caption}")
                    print("--------------------")

                except Exception as e:
                    print(f"\n⚠️ Warning: Failed to autocaption {vid_name}{ext}")
                    print(f"Error: {str(e)}")
                    
                    # Try alternate approach with still image
                    print("Trying alternate captioning approach with still image...")
                    final_caption = None
                    
                    # Extract a single frame to use as a still image
                    cap = cv2.VideoCapture(abs_path)
                    if cap.isOpened():
                        # Skip to middle of video
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                        ret, frame = cap.read()
                        if ret:
                            # Save frame as temporary image
                            temp_img_path = "temp_frame.jpg"
                            cv2.imwrite(temp_img_path, frame)
                            
                            try:
                                # Create message with image instead of video
                                image_messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image",
                                                "image": temp_img_path,
                                            },
                                            {
                                                "type": "text",
                                                "text": "Describe this scene briefly.",
                                            },
                                        ],
                                    }
                                ]
                                
                                text = processor.apply_chat_template(
                                    image_messages, tokenize=False, add_generation_prompt=True
                                )
                                
                                image_inputs, _ = process_vision_info(image_messages)
                                
                                if image_inputs and len(image_inputs) > 0:
                                    inputs = processor(
                                        text=[text],
                                        images=image_inputs,
                                        padding=True,
                                        return_tensors="pt",
                                    ).to("cuda")
                                    
                                    generated_ids = model.generate(
                                        **inputs,
                                        max_new_tokens=64,
                                        do_sample=True,
                                        temperature=0.7,
                                        top_p=0.9,
                                    )
                                    
                                    generated_ids_trimmed = [
                                        out_ids[len(in_ids):]
                                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                    ]
                                    
                                    caption = processor.batch_decode(
                                        generated_ids_trimmed,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False,
                                    )[0]
                                    
                                    final_caption = f"{prefix}{trigger}{caption.strip()}{suffix}"
                                    print("\n📝 Generated Caption (from still image):")
                                    print("--------------------")
                                    print(f"{final_caption}")
                                    print("--------------------")
                            except Exception as e2:
                                print(f"Still image captioning failed: {e2}")
                            
                            # Clean up temp file
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)
                    
                    # Use fallback caption if other methods failed
                    if not final_caption:
                        final_caption = f"{prefix}{trigger}A video clip named {vid_name}{suffix}"
                        print("\n📝 Using fallback caption:")
                        print("--------------------")
                        print(f"{final_caption}")
                        print("--------------------")

                # Save caption
                txt_path = os.path.join(videos_path, vid_name + ".txt")
                with open(txt_path, "w") as f:
                    f.write(final_caption.strip() + "\n")
                new_caption_files.add(vid_name)
                print(f"✅ Caption saved to: {txt_path}")
                break  # Found a video file, no need to check other extensions
        
        if not video_found:
            print(f"⚠️ Could not find video file for {vid_name}")

    # Clean up QWEN model
    print("\n=== 🧹 Cleaning Up GPU Memory ===")
    del model
    del processor
    torch.cuda.empty_cache()

    print(f"✨ Successfully processed {len(videos_without_captions)} videos!")
    print("=====================================")

    return new_caption_files


def create_dataset_toml(video_clip_mode: str) -> None:
    """Create dataset configuration file."""
    print("\n=== 📝 Creating Dataset Configuration ===")
    
    dataset_config = {
        "max_workers": 4,
        "size_buckets": [[512, 512, 16]],  # Width, Height, Frames
        "resolutions": [512],
        "cache_batch_size": 1,
        "frame_buckets": [1, 16],
        "video_clip_mode": video_clip_mode,
        "directory": [{
            "path": f"./{INPUT_DIR}/wan_train_replicate",
            "video_clip_length": 16,
            "num_repeats": 1
        }]
    }
    
    with open("wan_dataset.toml", "w") as f:
        toml.dump(dataset_config, f)
    
    print(f"✅ Dataset configuration created: wan_dataset.toml")
    print("=====================================")


def create_config_toml(
    model_type: str,
    finetuning_type: str,
    video_clip_mode: str,
    training_steps: int,
    learning_rate: float,
    lora_rank: int,
    warmup_steps: int,
    weight_decay: float,
    seed: int,
    num_gpus: int = 1
) -> None:
    """Create training configuration file with specified parameters."""
    print("\n=== 📝 Creating Training Configuration ===")
    
    # Select the appropriate model path based on model type and finetuning type
    if finetuning_type == "image2video":
        if model_type.lower() != "14b":
            raise ValueError("Image to video finetuning requires the 14B model.")
        model_path = os.path.join(MODEL_CACHE, "Wan2.1-I2V-14B-480P")
        print(f"Using I2V model for image-to-video training: {model_path}")
    else:  # text2video
        if model_type.lower() == "1.3b":
            model_path = os.path.join(MODEL_CACHE, "Wan2.1-T2V-1.3B")
        else:
            model_path = os.path.join(MODEL_CACHE, "Wan2.1-T2V-14B")
        print(f"Using T2V model for text-to-video training: {model_path}")
    
    # Adjust pipeline stages based on number of GPUs
    pipeline_stages = num_gpus if num_gpus > 1 else 1
    
    config = {
        "epochs": 999999,
        "max_steps": training_steps,
        "save_every_n_epochs": 999999,
        "checkpoint_every_n_minutes": 1440,
        "dataset": "wan_dataset.toml",
        "output_dir": f"./{OUTPUT_DIR}",
        "logging_steps": 10,
        "video_clip_mode": video_clip_mode,
        "pipeline_stages": pipeline_stages,  # Set pipeline stages to number of GPUs
        "general": {
            "train_batch_size": 1,
            "train_on_subset": True,
            "train_subset_size": 1,
            "checkpoint_frequency": 5
        },
        "eval_datasets": [],
        "model": {
            "type": "wan",
            "ckpt_path": model_path,
            "dtype": "bfloat16",
            "timestep_sample_method": "logit_normal"
        },
        "optimizer": {
            "type": "adamw",
            "lr": learning_rate,
            "betas": [0.9, 0.999],
            "weight_decay": weight_decay,
            "eps": 1e-8
        },
        "lr_scheduler": {
            "type": "cosine",
            "warmup_steps": warmup_steps
        },
        "adapter": {
            "type": "lora",
            "rank": lora_rank
        },
        "deepspeed": {
            "gradient_accumulation_steps": 1,
            "pipeline_stages": pipeline_stages,  # Set pipeline stages to number of GPUs
            "train_micro_batch_size_per_gpu": 1
        }
    }
    
    with open("wan_train_replicate.toml", "w") as f:
        toml.dump(config, f)
    
    print(f"✅ Training configuration created: wan_train_replicate.toml")
    print(f"  • Hard stopping at {training_steps} steps (once train.py checks max_steps).")
    print(f"  • Using {pipeline_stages} pipeline stages across {num_gpus} GPUs.")
    print("=====================================")


def run_training(training_steps: int, num_gpus: int = 1) -> None:
    """Execute the training process."""
    print("\n=== 🚀 Starting WAN Training ===")
    
    train_cmd = [
        "NCCL_P2P_DISABLE=1", 
        "NCCL_IB_DISABLE=1", 
        "deepspeed", 
        f"--num_gpus={num_gpus}", 
        "train.py", 
        "--config", 
        "wan_train_replicate.toml"
    ]
    
    cmd = " ".join(train_cmd)
    
    print(f"Training model with max {training_steps} steps on {num_gpus} GPUs")
    print(f"Command: {cmd}")
    print("Starting training process...")
    
    # Execute the training command
    process = subprocess.run(cmd, shell=True, check=True)
    
    # Verify training completed successfully
    if process.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"⚠️ Training exited with code: {process.returncode}")
    
    print("=====================================")


def archive_results() -> str:
    """Package training results and return path to archive."""
    print("\n=== 📦 Archiving Results ===")
    output_path = "trained_model.tar" 
    
    # Find the most recent training directory
    subdirs = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    
    if not subdirs:
        raise ValueError(f"No training directories found in {OUTPUT_DIR}")
    
    latest_dir = os.path.join(OUTPUT_DIR, subdirs[-1])
    print(f"Found training directory: {latest_dir}")
    
    # Find the latest checkpoint (epoch) directory
    epoch_dirs = [d for d in os.listdir(latest_dir) if d.startswith("epoch")]
    
    if not epoch_dirs:
        raise ValueError(f"No epoch directories found in {latest_dir}")
    
    # Get the highest epoch number
    highest_epoch = max(epoch_dirs, key=lambda x: int(x[5:]))
    adapter_dir = os.path.join(latest_dir, highest_epoch)
    
    print(f"Using latest checkpoint: {highest_epoch}")
    
    if not os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
        raise ValueError(f"No adapter model found at {adapter_dir}")
    
    # Create directory structure to exactly match the example
    # First, create a temporary directory with the exact structure we want
    temp_root = "temp_output_dir"
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    
    # Create the example directory structure
    example_output_dir = os.path.join(temp_root, "output")
    example_job_dir = os.path.join(example_output_dir, "wan_train_replicate")
    captions_dir = os.path.join(example_job_dir, "captions")
    
    # Create all directories
    os.makedirs(captions_dir, exist_ok=True)
    
    # Determine the model type from the config file
    with open("wan_train_replicate.toml", "r") as f:
        config = toml.load(f)
    
    # Extract model type from path
    model_path = config['model']['ckpt_path']
    model_type = "1.3b" if "1.3B" in model_path else "14b"
    is_i2v = "I2V" in model_path
    
    # Create appropriate filename based on type
    if is_i2v:
        lora_filename = f"{model_type}-i2v-lora.safetensors"
    else:
        lora_filename = f"{model_type}-lora.safetensors"
    
    # Copy and rename the safetensors file
    print(f"Copying adapter model to output structure as {lora_filename}")
    shutil.copy(
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(example_job_dir, lora_filename)
    )
    
    # Copy caption files
    caption_files = list(Path(INPUT_DIR).glob("wan_train_replicate/**/*.txt"))
    print(f"Adding {len(caption_files)} caption files to archive")
    for caption_file in caption_files:
        shutil.copy(caption_file, captions_dir)
    
    # Create the tar archive using the temp structure
    print(f"Creating final archive at {output_path}")
    os.system(f"tar -cvf {output_path} -C {temp_root} output")
    
    # Clean up temp directory after tar is created
    shutil.rmtree(temp_root)
    
    # Verify the file exists
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f"✅ Archive created successfully")
        print(f"  • Path: {output_path}")
        print(f"  • Size: {file_size_mb:.2f} MB")
        print(f"  • LoRA: {lora_filename}")
    else:
        print(f"⚠️ WARNING: Output file not found at {output_path}")
    
    print("=====================================")
    
    return output_path


def add_trigger_word_to_captions(trigger_word: str) -> None:
    """Add trigger word to caption files if not already present."""
    if not trigger_word:
        print("No trigger word provided, skipping caption modification")
        return
    
    # Find all caption files in the wan_train_replicate directory
    caption_path = f"{INPUT_DIR}/wan_train_replicate"
    caption_files = []
    for root, _, files in os.walk(caption_path):
        for file in files:
            if file.endswith(".txt"):
                caption_files.append(os.path.join(root, file))
    
    print(f"Found {len(caption_files)} caption files")
    
    # Process each caption file
    modified_count = 0
    for caption_file in caption_files:
        with open(caption_file, 'r') as f:
            content = f.read().strip()
        
        # Check if trigger word is already in the content
        if trigger_word not in content:
            # Add trigger word to the end of the content
            modified_content = f"{content} {trigger_word}"
            
            # Write modified content back to file
            with open(caption_file, 'w') as f:
                f.write(modified_content)
            
            modified_count += 1
            print(f"Added trigger word '{trigger_word}' to the end of {os.path.basename(caption_file)}")
    
    print(f"Modified {modified_count} caption files to include trigger word '{trigger_word}'")
