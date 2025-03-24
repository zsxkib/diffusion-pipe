# Required imports
import torch
import subprocess
import re

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


def print_gpu_memory_usage():
    """Print detailed memory usage for all available GPUs."""
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return

    print("\n=== 🔍 Current GPU Memory Usage ===")
    for i in range(torch.cuda.device_count()):
        # Get total memory in GB
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        # Get reserved memory in GB (allocated + cached)
        reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
        # Get allocated memory in GB (actively used)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        # Calculate utilization percentage
        utilization = (reserved_memory / total_memory) * 100
        
        print(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    - Total: {total_memory:.2f} GB")
        print(f"    - Reserved: {reserved_memory:.2f} GB ({utilization:.1f}%)")
        print(f"    - Allocated: {allocated_memory:.2f} GB")
    print("=====================================")


def get_nvidia_smi_info():
    """Get detailed GPU process information using nvidia-smi."""
    try:
        # Run nvidia-smi command to get process information
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stderr:
            print("Error getting nvidia-smi info:", result.stderr)
            return
            
        processes = result.stdout.strip().split('\n')
        if not processes or processes[0] == '':
            print("No GPU processes found")
            return
            
        print("\n=== 🔎 GPU Process Information ===")
        # Get GPU names
        gpu_names = {}
        for i in range(torch.cuda.device_count()):
            gpu_names[i] = torch.cuda.get_device_name(i)
            
        # Group processes by GPU
        gpu_processes = {}
        for process in processes:
            if not process.strip():
                continue
                
            parts = process.strip().split(', ')
            if len(parts) < 3:
                continue
                
            pid = parts[0]
            gpu_id = None
            
            # Try to map the GPU UUID to a device index
            try:
                uuid_result = subprocess.run(
                    ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                for i in range(torch.cuda.device_count()):
                    uuid_check = subprocess.run(
                        ["nvidia-smi", "-i", str(i), "--query-gpu=uuid", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    if uuid_check.stdout.strip() == parts[1]:
                        gpu_id = i
                        break
            except:
                pass
                
            if gpu_id is None:
                # If we couldn't map it, just use a counter
                gpu_id = parts[1][-1] if parts[1] else "?"
                
            mem_used = parts[2]
            
            if gpu_id not in gpu_processes:
                gpu_processes[gpu_id] = []
                
            # Get command for this PID
            try:
                cmd_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "command="],
                    capture_output=True,
                    text=True,
                    check=True
                )
                cmd = cmd_result.stdout.strip()
                # Truncate command if too long
                if len(cmd) > 80:
                    cmd = cmd[:77] + "..."
            except:
                cmd = "Unknown"
                
            gpu_processes[gpu_id].append((pid, mem_used, cmd))
            
        # Print process information by GPU
        for gpu_id, processes in gpu_processes.items():
            if isinstance(gpu_id, int):
                print(f"  • GPU {gpu_id}: {gpu_names.get(gpu_id, 'Unknown GPU')}")
            else:
                print(f"  • GPU {gpu_id}")
                
            for pid, mem_used, cmd in processes:
                print(f"    - PID {pid}: {mem_used} ({cmd})")
                
        print("=====================================")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        # Fallback to basic memory information
        print_gpu_memory_usage()