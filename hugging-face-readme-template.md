---
license: apache-2.0
language:
- en
- zh
tags:
- image-to-video
- lora
- replicate
- text-to-video
- video
- video-generation
base_model: "Wan-AI/Wan2.1-${t2v_or_i2v}2V-${model_type}-Diffusers"
pipeline_tag: ${pipeline_tag}
# widget:
#   - text: >-
#       prompt
#     output:
#       url: https://...
$instance_prompt
---

# $title

<Gallery />

## About this LoRA

This is a [LoRA](https://replicate.com/docs/guides/working-with-loras) for the Wan ${model_type} ${readable_finetuning_type} model.

It can be used with diffusers or ComfyUI, and can be loaded against the Wan ${model_type} models.

It was trained on [Replicate](https://replicate.com/) with ${max_training_steps} steps at a learning rate of ${learning_rate} and LoRA rank of ${lora_rank}.

$trigger_section

## Use this LoRA

Replicate has a collection of Wan models that are optimised for speed and cost. They can also be used with this LoRA:

- https://replicate.com/collections/wan-video
- https://replicate.com/fofr/wan-with-lora

### Run this LoRA with an API using Replicate

```py
import replicate

input = {
    "prompt": "$trigger_word",
    "lora_url": "https://huggingface.co/$repo_id/resolve/main/$lora_filename.safetensors"
}

output = replicate.run(
    "fofr/wan-with-lora:latest",
    model="${model_type}",
    input=input
)
for index, item in enumerate(output):
    with open(f"output_{index}.mp4", "wb") as file:
        file.write(item.read())
```

### Using with Diffusers

```py
import torch
from diffusers.utils import export_to_video
from diffusers import WanVidAdapter, WanVid

# Load base model
base_model = WanVid.from_pretrained("Wan-AI/Wan2.1-${t2v_or_i2v}2V-${model_type}-Diffusers", torch_dtype=torch.float16)

# Load and apply LoRA adapter
adapter = WanVidAdapter.from_pretrained("$repo_id")
base_model.load_adapter(adapter)

# Generate video
prompt = "$trigger_word"
negative_prompt = "blurry, low quality, low resolution"

# Generate video frames
$generation_code

# Save as video
video_path = "output.mp4"
export_to_video(frames, video_path, fps=16)
print(f"Video saved to: {video_path}")
```

$training_details

## Contribute your own examples

You can use the [community tab](https://huggingface.co/$repo_id/discussions) to add videos that show off what you've made with this LoRA.
