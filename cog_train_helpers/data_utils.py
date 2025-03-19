# Required imports
import os
import shutil
import time
import cv2
import torch
from zipfile import ZipFile, is_zipfile
from pathlib import Path
from typing import Optional, Set
from cog import Path as CogPath
from qwen_vl_utils import process_vision_info

# Import from other modules
from cog_train_helpers.constants import INPUT_DIR, JOB_NAME, QWEN_MODEL_CACHE
from cog_train_helpers.model_utils import download_weights, setup_qwen_model

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
