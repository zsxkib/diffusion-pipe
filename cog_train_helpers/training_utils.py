# Required imports
import os
import shutil
import subprocess
import toml
from pathlib import Path
import tarfile

# Import constants
from cog_train_helpers.constants import INPUT_DIR, OUTPUT_DIR, QWEN_MODEL_CACHE

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


def archive_results():
    """Archive training results into a tar file and prepare HF upload directory."""
    print("\n=== 📦 Archiving Results ===")
    
    # Find the latest training directory
    training_dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    
    if not training_dirs:
        raise ValueError(f"No training directories found in {OUTPUT_DIR}")
    
    latest_dir = os.path.join(OUTPUT_DIR, training_dirs[-1])
    
    # Find highest epoch directory
    epoch_dirs = [d for d in os.listdir(latest_dir) if d.startswith("epoch")]
    highest_epoch = max(epoch_dirs, key=lambda x: int(x[5:]))
    adapter_dir = os.path.join(latest_dir, highest_epoch)
    
    # Setup HF upload directory
    hf_upload_dir = os.path.join("output", "wan_train_replicate")
    captions_dir = os.path.join(hf_upload_dir, "captions")
    
    # Clear and recreate directory
    if os.path.exists(hf_upload_dir):
        shutil.rmtree(hf_upload_dir)
    
    os.makedirs(captions_dir, exist_ok=True)
    
    # Get model type from config
    with open("wan_train_replicate.toml", "r") as f:
        config = toml.load(f)
    
    model_path = config['model']['ckpt_path']
    model_type = "1.3b" if "1.3B" in model_path else "14b"
    is_i2v = "I2V" in model_path
    
    # Create appropriate filename
    lora_filename = f"{model_type}-{'i2v' if is_i2v else 'lora'}.safetensors"
    
    # Copy safetensors file
    shutil.copy(
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(hf_upload_dir, lora_filename)
    )
    
    # Copy captions
    caption_files = list(Path(INPUT_DIR).glob("wan_train_replicate/**/*.txt"))
    for caption_file in caption_files:
        shutil.copy(caption_file, captions_dir)
    
    # Create tar archive
    output_path = "trained_model.tar"
    with tarfile.open(output_path, "w") as tar:
        tar.add(hf_upload_dir, arcname=os.path.basename(hf_upload_dir))
    
    print(f"✅ Archive created at {output_path}")
    
    return output_path

