# Required imports
import os
import toml
from cog_train_helpers.constants import MODEL_CACHE, INPUT_DIR, OUTPUT_DIR

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
    num_gpus: int = 1,
    wandb_sample_interval: int = None,
    wandb_save_interval: int = None
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
    
    # Add sample_every if W&B sample interval is provided
    if wandb_sample_interval:
        # Create the sample section if it doesn't exist
        if "sample" not in config:
            config["sample"] = {}
        config["sample"]["sample_every"] = wandb_sample_interval
        print(f"  • Sample visualization every {wandb_sample_interval} steps")
    
    # Add save_every if W&B save interval is provided
    if wandb_save_interval:
        # Create the save section if it doesn't exist
        if "save" not in config:
            config["save"] = {}
        config["save"]["save_every"] = wandb_save_interval
        print(f"  • Model checkpoint saved every {wandb_save_interval} steps")
    
    with open("wan_train_replicate.toml", "w") as f:
        toml.dump(config, f)
    
    print(f"✅ Training configuration created: wan_train_replicate.toml")
    print(f"  • Hard stopping at {training_steps} steps (once train.py checks max_steps).")
    print(f"  • Using {pipeline_stages} pipeline stages across {num_gpus} GPUs.")
    print("=====================================")

