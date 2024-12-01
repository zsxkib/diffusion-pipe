from pathlib import Path
import json
import sys
import os.path
sys.path.insert(0, os.path.abspath('submodules/LTX-Video'))

from transformers import T5EncoderModel, T5Tokenizer
import safetensors
import imageio
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from PIL import ImageOps
from torchvision import transforms

from models.base import BasePipeline
from utils.common import VIDEO_EXTENSIONS
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline as OriginalLTXVideoPipeline
from ltx_video.models.autoencoders.vae_encode import vae_encode


FRAMERATE = 25


def load_vae(vae_dir, dtype):
    vae_ckpt_path = vae_dir / 'vae_diffusion_pytorch_model.safetensors'
    vae_config_path = vae_dir / 'config.json'
    with open(vae_config_path, 'r') as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    return vae.to(dtype)


def load_unet(unet_dir, dtype):
    unet_ckpt_path = unet_dir / 'unet_diffusion_pytorch_model.safetensors'
    unet_config_path = unet_dir / 'config.json'
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    return transformer.to(dtype)


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / 'scheduler_config.json'
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)


def extract_clips(video, target_frames, config):
    # video is (channels, num_frames, height, width)
    frames = video.shape[1]
    if frames < target_frames:
        # TODO: think about how to handle this case. Maybe the video should have already been thrown out?
        print(f'video with shape {video.shape} is being skipped because it has less than the target_frames')
        return []

    video_clip_mode = config.get('video_clip_mode', 'single_middle')
    if video_clip_mode == 'single_beginning':
        return [video[:, :target_frames, ...]]
    elif video_clip_mode == 'single_middle':
        start = int((frames - target_frames) / 2)
        assert frames-start >= target_frames
        return [video[:, start:start+target_frames, ...]]
    elif video_clip_mode == 'multiple_overlapping':
        # Extract multiple clips so we use the whole video for training.
        # The clips might overlap a little bit. We never cut anything off the end of the video.
        num_clips = ((frames - 1) // target_frames) + 1
        start_indices = torch.linspace(0, frames-target_frames, num_clips).int()
        return [video[:, i:i+target_frames, ...] for i in start_indices]
    else:
        raise NotImplementedError(f'video_clip_mode={video_clip_mode} is not recognized')


class PreprocessMediaFile:
    def __init__(self, config):
        self.config = config
        self.pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __call__(self, filepath, size_bucket):
        width, height, frames = size_bucket
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        frames_padded = ((frames - 2) // 8 + 1) * 8 + 1

        is_video = (Path(filepath).suffix in VIDEO_EXTENSIONS)
        if is_video:
            num_frames = 0
            for frame in imageio.v3.imiter(filepath, fps=FRAMERATE):
                channels = frame.shape[-1]
                num_frames += 1
            frames = imageio.v3.imiter(filepath, fps=FRAMERATE)
        else:
            num_frames = 1
            frames = [imageio.v3.imread(filepath)]
            channels = frames[0].shape[-1]

        video = torch.empty((num_frames, channels, height_padded, width_padded))
        for i, frame in enumerate(frames):
            pil_image = torchvision.transforms.functional.to_pil_image(frame)
            cropped_image = ImageOps.fit(pil_image, (width_padded, height_padded))
            video[i, ...] = self.pil_to_tensor(cropped_image)

        # (num_frames, channels, height, width) -> (channels, num_frames, height, width)
        video = torch.permute(video, (1, 0, 2, 3))

        return extract_clips(video, frames_padded, self.config)


class LTXVideoPipeline(BasePipeline):
    name = 'ltx-video'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['BasicTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']

        dtype = self.model_config['dtype']
        ckpt_dir = Path(self.model_config['diffusers_path'])
        vae = load_vae(ckpt_dir / 'vae', dtype)
        unet = load_unet(ckpt_dir / 'unet', dtype)
        scheduler = load_scheduler(ckpt_dir / 'scheduler')
        patchifier = SymmetricPatchifier(patch_size=1)
        text_encoder = T5EncoderModel.from_pretrained(ckpt_dir, subfolder='text_encoder', torch_dtype=dtype)
        tokenizer = T5Tokenizer.from_pretrained(ckpt_dir, subfolder='tokenizer', torch_dtype=dtype)
        submodel_dict = {
            'transformer': unet,
            'patchifier': patchifier,
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'scheduler': scheduler,
            'vae': vae,
        }
        self.diffusers_pipeline = OriginalLTXVideoPipeline(**submodel_dict)

        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return (self.text_encoder,)

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # Convention is to have "transformer." prefix
        peft_state_dict = {'transformer.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config)

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            return {'latents': vae_encode(tensor.to(vae.device, vae.dtype), vae, vae_per_channel_normalize=True)}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption):
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(caption, do_classifier_free_guidance=False, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds, 'prompt_attention_mask': prompt_attention_mask}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        prompt_attention_mask = inputs['prompt_attention_mask']

        bs, channels, num_frames, h, w = latents.shape
        latent_frame_rate = FRAMERATE / self.video_scale_factor
        latent_frame_rates = (
            torch.ones(
                bs, 1, device=latents.device
            )
            * latent_frame_rate
        )
        scale_grid = (
            (
                1 / latent_frame_rates,
                self.vae_scale_factor,
                self.vae_scale_factor,
            )
            if self.transformer.use_rope
            else None
        )
        latents = self.patchifier.patchify(latents)
        indices_grid = self.patchifier.get_grid(
            orig_num_frames=num_frames,
            orig_height=h,
            orig_width=w,
            batch_size=bs,
            scale_grid=scale_grid,
            device=latents.device,
        )

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        dtype = self.model_config['dtype']
        return x_t.to(dtype), indices_grid, prompt_embeds.to(dtype), prompt_attention_mask, t, target

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.transformer_blocks:
            layers.append(TransformerLayer(block))
        layers.append(OutputLayer(transformer))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]
        # Explicitly register these modules.
        self.patchify_proj = self.transformer[0].patchify_proj
        self.adaln_single = self.transformer[0].adaln_single
        self.caption_projection = self.transformer[0].caption_projection

    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        hidden_states, indices_grid, encoder_hidden_states, encoder_attention_mask, timestep, target = inputs

        if encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        hidden_states = self.patchify_proj(hidden_states)

        if self.transformer[0].timestep_scale_multiplier:
            timestep = self.transformer[0].timestep_scale_multiplier * timestep

        if self.transformer[0].positional_embedding_type == "absolute":
            pos_embed_3d = self.transformer[0].get_absolute_pos_embed(indices_grid).to(
                hidden_states.device
            )
            if self.transformer[0].project_to_2d_pos:
                pos_embed = self.transformer[0].to_2d_proj(pos_embed_3d)
            hidden_states = (hidden_states + pos_embed).to(hidden_states.dtype)
            freqs_cos, freqs_sin = None, None
        elif self.transformer[0].positional_embedding_type == "rope":
            freqs_cos, freqs_sin = self.transformer[0].precompute_freqs_cis(indices_grid)

        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        ret = hidden_states, freqs_cos, freqs_sin, encoder_hidden_states, encoder_attention_mask, timestep, embedded_timestep, target
        return ret


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden_states, freqs_cos, freqs_sin, encoder_hidden_states, encoder_attention_mask, timestep, embedded_timestep, target = inputs
        hidden_states = self.block(
            hidden_states,
            freqs_cis=(freqs_cos, freqs_sin),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
        )
        return hidden_states, freqs_cos, freqs_sin, encoder_hidden_states, encoder_attention_mask, timestep, embedded_timestep, target


class OutputLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]
        # Explicitly register these modules.
        self.scale_shift_table = self.transformer[0].scale_shift_table
        self.norm_out = self.transformer[0].norm_out
        self.proj_out = self.transformer[0].proj_out

    def forward(self, inputs):
        hidden_states, freqs_cos, freqs_sin, encoder_hidden_states, encoder_attention_mask, timestep, embedded_timestep, target = inputs
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        output = output.to(torch.float32)
        target = target.to(torch.float32)
        loss = F.mse_loss(output, target)
        return loss
