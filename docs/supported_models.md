# Summary

| Model         | LoRA | Full Fine Tune | fp8/quantization |
|---------------|------|----------------|------------------|
|SDXL           |✅    |❌              |❌                |
|Flux           |✅    |✅              |✅                |
|LTX-Video      |✅    |❌              |❌                |
|HunyuanVideo   |✅    |❌              |✅                |
|Cosmos         |✅    |❌              |❌                |


## SDXL
```
[model]
type = 'sdxl'
checkpoint_path = '/data2/imagegen_models/sdxl/sd_xl_base_1.0_0.9vae.safetensors'
dtype = 'bfloat16'
# You can train v-prediction models (e.g. NoobAI vpred) by setting this option.
#v_pred = true
# Min SNR is supported. Same meaning as sd-scripts
#min_snr_gamma = 5
```
Basic SDXL LoRA support is implemented. It is lacking many options present in other training scripts. For example, you currently can't set separate learning rates for text encoders and unet.

Unlike other models, text embeddings are not cached, and the text encoders are trained.

SDXL LoRAs are saved in Kohya sd-scripts format.

## Flux
```
[model]
type = 'flux'
# Path to Huggingface Diffusers directory for Flux
diffusers_path = '/data2/imagegen_models/FLUX.1-dev'
# You can override the transformer from a BFL format checkpoint.
#transformer_path = '/data2/imagegen_models/flux-dev-single-files/consolidated_s6700-schnell.safetensors'
dtype = 'bfloat16'
# Flux supports fp8 for the transformer when training LoRA.
transformer_dtype = 'float8'
# Resolution-dependent timestep shift towards more noise. Same meaning as sd-scripts.
flux_shift = true
# For FLEX.1-alpha, you can bypass the guidance embedding which is the recommended way to train that model.
#bypass_guidance_embedding = true
```
For Flux, you can override the transformer weights by setting transformer_path to an original Black Forest Labs (BFL) format checkpoint. For example, the above config loads the model from Diffusers format FLUX.1-dev, but the transformer_path, if uncommented, loads the transformer from Flux Dev De-distill.

Flux LoRAs are saved in Diffusers format.

## LTX-Video
```
[model]
type = 'ltx-video'
diffusers_path = '/data2/imagegen_models/LTX-Video'
dtype = 'bfloat16'
timestep_sample_method = 'logit_normal'
```
LTX-Video LoRAs are saved in Diffusers format.

## HunyuanVideo
```
[model]
type = 'hunyuan-video'
# Can load Hunyuan Video entirely from the ckpt path set up for the official inference scripts.
#ckpt_path = '/home/anon/HunyuanVideo/ckpts'
# Or you can load it by pointing to all the ComfyUI files.
transformer_path = '/data2/imagegen_models/hunyuan_video_comfyui/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors'
vae_path = '/data2/imagegen_models/hunyuan_video_comfyui/hunyuan_video_vae_bf16.safetensors'
llm_path = '/data2/imagegen_models/hunyuan_video_comfyui/llava-llama-3-8b-text-encoder-tokenizer'
clip_path = '/data2/imagegen_models/hunyuan_video_comfyui/clip-vit-large-patch14'
# Base dtype used for all models.
dtype = 'bfloat16'
# Hunyuan Video supports fp8 for the transformer when training LoRA.
transformer_dtype = 'float8'
# How to sample timesteps to train on. Can be logit_normal or uniform.
timestep_sample_method = 'logit_normal'
```
HunyuanVideo LoRAs are saved in a Diffusers-style format. The keys are named according to the original model, and prefixed with "transformer.". This format will directly work with ComfyUI.

## Cosmos
```
[model]
type = 'cosmos'
transformer_path = '/data2/imagegen_models/cosmos/cosmos-1.0-diffusion-7b-text2world.pt'
vae_path = '/data2/imagegen_models/cosmos/cosmos_cv8x8x8_1.0.safetensors'
text_encoder_path = '/data2/imagegen_models/cosmos/oldt5_xxl_fp16.safetensors'
dtype = 'bfloat16'
```
Tentative support is added for Cosmos (text2world diffusion variants). Compared to HunyuanVideo, Cosmos is not good for fine-tuning on commodity hardware.

1. Cosmos supports a fixed, limited set of resolutions and frame lengths. Because of this, the 7b model is actually slower to train than HunyuanVideo (12b parameters), because you can't get away with training on lower-resolution images like you can with Hunyuan. And video training is nearly impossible unless you have enormous amounts of VRAM, because for videos you must use the full 121 frame length.
2. Cosmos seems much worse at generalizing from image-only training to video.
3. The Cosmos base model is much more limited in the types of content that it knows, which makes fine tuning for most concepts more difficult.

I will likely not be actively supporting Cosmos going forward. All the pieces are there, and if you really want to try training it you can. But don't expect me to spend time trying to fix things if something doesn't work right.

Cosmos LoRAs are saved in ComfyUI format.