# Required imports
import torch

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