# Required imports
import os
import shutil
import subprocess
import toml
from pathlib import Path
import tarfile
import time
import re

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


def run_training(training_steps: int, num_gpus: int = 1, wandb_client=None) -> None:
    """Execute the training process with optional W&B logging."""
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
    
    # If W&B is enabled, capture output to extract metrics
    if wandb_client:
        # Run with real-time output capture
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Pattern to extract loss information
        step_pattern = re.compile(r"steps: (\d+) loss: ([\d\.]+)")
        
        # Initialize sample tracking
        last_sample_check = time.time()
        sample_check_interval = 60  # Check for new samples every 60 seconds
        seen_samples = set()
        current_step = 0
        
        # Process output line by line
        for line in iter(process.stdout.readline, ''):
            # Print the output to console
            print(line, end='')
            
            # Extract and log loss if available
            step_match = step_pattern.search(line)
            if step_match and wandb_client:
                current_step = int(step_match.group(1))
                loss = float(step_match.group(2))
                wandb_client.log_loss({"loss": loss}, current_step)
            
            # Periodically check for samples
            if wandb_client and time.time() - last_sample_check > sample_check_interval:
                last_sample_check = time.time()
                output_dir = os.path.join(OUTPUT_DIR, "samples")
                
                if os.path.exists(output_dir):
                    all_samples = set()
                    for ext in ['mp4', 'gif', 'png', 'jpg', 'jpeg']:
                        for sample_file in Path(output_dir).glob(f"*.{ext}"):
                            all_samples.add(str(sample_file))
                    
                    # Find new samples
                    new_samples = all_samples - seen_samples
                    if new_samples:
                        sample_paths = [Path(p) for p in sorted(new_samples)]
                        wandb_client.log_samples(sample_paths, current_step)
                        seen_samples.update(new_samples)
        
        # Wait for process to complete
        process.wait()
        return_code = process.returncode
    else:
        # If W&B not enabled, just run normally
        process = subprocess.run(cmd, shell=True, check=True)
        return_code = process.returncode
    
    # Verify training completed successfully
    if return_code == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"⚠️ Training exited with code: {return_code}")
    
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

