# Required imports
import os
import shutil
import subprocess
import toml
from pathlib import Path

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
    """Archive the training results into a tar file."""
    print("\n=== 📦 Archiving Results ===")
    
    # Find the latest training directory
    training_dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    
    if not training_dirs:
        raise ValueError(f"No training directories found in {OUTPUT_DIR}")
    
    latest_dir = os.path.join(OUTPUT_DIR, training_dirs[-1])
    print(f"Found training directory: {latest_dir}")
    
    # Find all epoch directories
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
    output_path = "trained_model.tar"
    print(f"Creating final archive at {output_path}")
    os.system(f"tar -cvf {output_path} -C {temp_root} output")
    
    # Create a persistent directory for HF uploads with the same content
    # This ensures we have a directory to upload even after the temp dir is cleaned up
    hf_upload_dir = os.path.join("output", "wan_train_replicate")
    os.makedirs(os.path.dirname(hf_upload_dir), exist_ok=True)
    
    # Copy directory to output/ for HF upload
    if os.path.exists(hf_upload_dir):
        shutil.rmtree(hf_upload_dir)
    shutil.copytree(example_job_dir, hf_upload_dir)
    
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

