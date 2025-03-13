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
import time
import toml
from pathlib import Path
from zipfile import ZipFile, is_zipfile
from cog import BaseModel, Input, Path as CogPath  # Removed Secret import
from typing import Optional
import logging

# Configure logging to suppress INFO messages
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Suppress common loggers
loggers_to_quiet = [
    "torch", "accelerate", "transformers", "__main__", "PIL",
    "safetensors", "xformers", "datasets", "tokenizers", 
    "diffusers", "filelock", "bitsandbytes"
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

def download_weights(url: str, dest: str) -> None:
    """Download weights from URL to destination path."""
    start = time.time()
    print("[!] Initiating download from URL:", url)
    print("[~] Destination path:", dest)
    
    if ".tar" in dest:
        dest = os.path.dirname(dest)
        
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. "
            f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        
    print("[!] Download took:", time.time() - start, "seconds")


def download_model(model_type: str = "1.3b") -> None:
    """Download model weights based on specified model type."""
    if model_type.lower() == "1.3b":
        # Download only the 1.3B T2V model
        model_files = ["Wan2.1-T2V-1.3B.tar"]
    elif model_type.lower() == "14b":
        # Download 14B models
        model_files = [
            "Wan2.1-T2V-14B.tar",
            "Wan2.1-I2V-14B-720P.tar",
            "Wan2.1-I2V-14B-480P.tar",
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose '1.3b' or '14b'.")
    
    if not os.path.exists(MODEL_CACHE):
        os.makedirs(MODEL_CACHE)
    
    for model_file in model_files:
        url = BASE_URL + model_file
        filename = url.split("/")[-1]
        dest_path = os.path.join(MODEL_CACHE, filename)
        
        if not os.path.exists(dest_path.replace(".tar", "")):
            print(f"Downloading {model_file}...")
            download_weights(url, dest_path)
        else:
            print(f"Model {model_file} already exists, skipping download")
    
    print(f"✅ Model download check completed for {model_type} model(s)")

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
    video_clip_mode: str = Input(
        description="How to use your video during training: 'single_beginning' (focus on the opening scene), 'single_middle' (focus on the middle portion, ignoring intro/outro), or 'multiple_overlapping' (try to learn from the entire video by using multiple segments). For most cases, 'single_middle' gives the best results.",
        default="single_middle",
        choices=["single_beginning", "single_middle", "multiple_overlapping"],
    ),
    trigger_word: str = Input(
        description="The trigger word to be associated with all videos during training. This word will help activate the LoRA when used in prompts.",
        default="TOK",
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

    print("\n=== 🎥 WAN Video LoRA Training ===")
    print("📊 Configuration:")
    print(f"  • Input: {input_video_zip}")
    print(f"  • Model Type: {model_type}")
    print(f"  • Max Training Steps: {max_training_steps}")
    print(f"  • Warmup Steps Budget: {warmup_steps_budget}")
    print(f"  • LoRA Rank: {lora_rank}")
    print(f"  • Learning Rate: {learning_rate}")
    print("=====================================\n")

    if not input_video_zip:
        raise ValueError("You must provide a zip file containing training data.")

    if not is_zipfile(input_video_zip):
        raise ValueError("The provided input must be a zip file.")

    # Setup directories and seed
    clean_up()
    seed = handle_seed(seed)
    
    # Download the model with specified type
    download_model(model_type)
    
    # Extract zip and set up data directory
    extract_zip(input_video_zip, INPUT_DIR)
    
    # Add trigger word to captions
    add_trigger_word_to_captions(trigger_word)
    
    # Create configuration files
    create_dataset_toml(video_clip_mode)
    create_config_toml(
        model_type=model_type,
        video_clip_mode=video_clip_mode,
        training_steps=max_training_steps,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        warmup_steps=warmup_steps_budget,
        weight_decay=weight_decay,
        seed=seed
    )
    
    # Run training - pass max_training_steps to force a hard stop
    run_training(max_training_steps)
    
    # Archive results
    output_path = archive_results()
    
    # Simple output with just the weights
    return TrainingOutput(weights=CogPath(output_path))


def handle_seed(seed: int) -> int:
    """Set up random seed, generating one if seed is -1."""
    if seed == -1:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    return seed


def clean_up() -> None:
    """Clean up existing directories before training."""
    # Clean directories
    for dir in [INPUT_DIR, OUTPUT_DIR]:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def extract_zip(zip_path: CogPath, input_dir: str) -> None:
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
            
            # Move to the target directory with a flat structure
            dest_path = os.path.join(target_dir, file)
            
            # If file with same name exists, add a unique suffix
            if os.path.exists(dest_path):
                filename, ext = os.path.splitext(file)
                dest_path = os.path.join(target_dir, f"{filename}_{int(time.time())}{ext}")
            
            shutil.copy2(file_path, dest_path)
            
            # Track by file type
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.add(base_name)
                video_count += 1
            elif file.lower().endswith('.txt'):
                caption_files.add(base_name)
                text_count += 1
    
    # Clean up temporary directory
    shutil.rmtree(temp_extract_dir)
    
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
    video_clip_mode: str,
    training_steps: int,
    learning_rate: float,
    lora_rank: int,
    warmup_steps: int,
    weight_decay: float,
    seed: int
) -> None:
    """Create training configuration file with specified parameters."""
    print("\n=== 📝 Creating Training Configuration ===")
    
    if model_type.lower() == "1.3b":
        model_path = os.path.join(MODEL_CACHE, "Wan2.1-T2V-1.3B")
    else:
        model_path = os.path.join(MODEL_CACHE, "Wan2.1-T2V-14B")
    
    config = {
        "epochs": 999999,
        "max_steps": training_steps,
        "save_every_n_epochs": 999999,
        "checkpoint_every_n_minutes": 1440,
        "dataset": "wan_dataset.toml",
        "output_dir": f"./{OUTPUT_DIR}",
        "logging_steps": 10,
        "video_clip_mode": video_clip_mode,
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
            "pipeline_stages": 1,
            "train_micro_batch_size_per_gpu": 1
        }
    }
    
    with open("wan_train_replicate.toml", "w") as f:
        toml.dump(config, f)
    
    print(f"✅ Training configuration created: wan_train_replicate.toml")
    print(f"  • Hard stopping at {training_steps} steps (once train.py checks max_steps).")
    print("=====================================")


def run_training(training_steps: int) -> None:
    """Execute the training process."""
    print("\n=== 🚀 Starting WAN Training ===")
    
    train_cmd = [
        "NCCL_P2P_DISABLE=1", 
        "NCCL_IB_DISABLE=1", 
        "deepspeed", 
        "--num_gpus=1", 
        "train.py", 
        "--config", 
        "wan_train_replicate.toml"
    ]
    
    cmd = " ".join(train_cmd)
    
    print(f"Running command: {cmd}")
    print(f"⚠️ Training will run for exactly {training_steps} steps (via max_steps parameter)")
    
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
    
    # Find the latest checkpoint (epoch) directory
    epoch_dirs = [d for d in os.listdir(latest_dir) if d.startswith("epoch")]
    
    if not epoch_dirs:
        raise ValueError(f"No epoch directories found in {latest_dir}")
    
    # Get the highest epoch number
    highest_epoch = max(epoch_dirs, key=lambda x: int(x[5:]))
    adapter_dir = os.path.join(latest_dir, highest_epoch)
    
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
    
    # Copy and rename the safetensors file
    shutil.copy(
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(example_job_dir, "lora.safetensors")
    )
    
    # Copy caption files
    caption_files = list(Path(INPUT_DIR).glob("wan_train_replicate/**/*.txt"))
    for caption_file in caption_files:
        shutil.copy(caption_file, captions_dir)
    
    # Create the tar archive using the temp structure
    print(f"Archiving adapter outputs to {output_path}")
    os.system(f"tar -cvf {output_path} -C {temp_root} output")
    
    # Clean up temp directory after tar is created
    shutil.rmtree(temp_root)
    
    # Verify the file exists
    if os.path.exists(output_path):
        print(f"✅ Verified file exists at: {output_path}")
        # Get file size for sanity check
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
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
        try:
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
        except Exception as e:
            print(f"Error processing caption file {caption_file}: {str(e)}")
    
    print(f"Modified {modified_count} caption files to include trigger word '{trigger_word}'")
