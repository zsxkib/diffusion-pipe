# Required imports
import os
import shutil
import subprocess
import toml
import time
import signal
import sys
import threading
from pathlib import Path

# Import constants
from cog_train_helpers.constants import INPUT_DIR, OUTPUT_DIR, QWEN_MODEL_CACHE
from cog_train_helpers.gpu_utils import print_gpu_memory_usage, get_nvidia_smi_info

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
    """Execute the training process with periodic GPU monitoring."""
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
    
    # Print initial GPU state before training starts
    print_gpu_memory_usage()
    
    # Execute the training command as a subprocess that we can monitor
    process = subprocess.Popen(cmd, shell=True)
    
    # Sleep briefly to let the process start and allocate GPU resources
    time.sleep(10)
    
    # Check if the process has already exited (failed to start)
    if process.poll() is not None:
        print(f"⚠️ Process exited immediately with code: {process.returncode}")
        return
    
    # After the process starts, show the initial GPU usage with process information
    print("\n=== 🚀 Initial Training GPU Status ===")
    get_nvidia_smi_info()
    
    # Set up monitoring interval (every 100 steps, approximated by time)
    # Estimate time per step - this is a rough estimate and can be adjusted
    estimated_seconds_per_step = 3  # assuming ~3 seconds per step on average
    check_interval = min(100 * estimated_seconds_per_step, 300)  # Check at least every 5 minutes
    
    # Monitor the GPU usage while the training is running
    stop_monitoring = False
    
    def monitor_gpu_usage():
        last_print_time = 0
        step_counter = 0
        while not stop_monitoring:
            current_time = time.time()
            if current_time - last_print_time >= check_interval:
                step_counter += 100  # Approximate steps
                print(f"\n=== Step ~{step_counter} GPU Monitoring ===")
                # Use both monitoring methods
                print_gpu_memory_usage()
                get_nvidia_smi_info()
                last_print_time = current_time
            time.sleep(10)  # Check every 10 seconds if it's time to print
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_gpu_usage)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Wait for the process to complete
        exit_code = process.wait()
        
        # Stop the monitoring thread
        stop_monitoring = True
        monitor_thread.join(timeout=2)
        
        # Print final GPU state after training
        print("\n=== 🏁 Final GPU Status After Training ===")
        print_gpu_memory_usage()
        get_nvidia_smi_info()
        
        # Check if training completed successfully
        if exit_code == 0:
            print("✅ Training completed successfully!")
        else:
            print(f"⚠️ Training exited with code: {exit_code}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        process.send_signal(signal.SIGINT)
        stop_monitoring = True
        monitor_thread.join(timeout=2)
        print_gpu_memory_usage()
        get_nvidia_smi_info()
        
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

