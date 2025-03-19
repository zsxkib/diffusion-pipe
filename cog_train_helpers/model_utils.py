# Required imports
import os
import time
import subprocess
import shutil
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Import constants
from cog_train_helpers.constants import MODEL_CACHE, QWEN_MODEL_CACHE, QWEN_MODEL_URL, BASE_URL

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

