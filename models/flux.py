import math

import diffusers
import peft
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from PIL import Image, ImageOps
from deepspeed.utils.logging import logger
from safetensors import safe_open

from utils.common import is_main_process

ADAPTER_TARGET_MODULES = ['FluxTransformerBlock', 'FluxSingleTransformerBlock']


def crop_and_resize(pil_img, size_bucket):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    return ImageOps.fit(pil_img, size_bucket)


pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
def encode_pil_to_latents(pil_imgs, vae):
    img = torch.stack([pil_to_tensor(pil_img) for pil_img in pil_imgs])
    latents = vae.encode(img.to(vae.device, vae.dtype)).latent_dist.sample()
    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
        latents = latents - vae.config.shift_factor
    latents = latents * vae.config.scaling_factor
    latents = latents.to('cpu')
    return latents


tensor_to_pil = transforms.Compose([transforms.Lambda(lambda x: (x / 2 + 0.5).clamp(0, 1)), transforms.ToPILImage()])
def decode_latents_to_pil(latents, vae):
    latents = latents.to(vae.device)
    latents = latents / vae.config.scaling_factor
    if hasattr(vae.config, 'shift_factor'):
        latents = latents + vae.config.shift_factor
    img = vae.decode(latents.to(vae.dtype), return_dict=False)[0].to(torch.float32)
    img = img.squeeze(0)
    return tensor_to_pil(img)


def process_image_fn(vae, size_bucket):
    def fn(example):
        pil_imgs = []
        for image_file in example['image_file']:
            try:
                pil_img = Image.open(image_file)
            except Exception:
                logger.warning(f'Image file {image_file} could not be opened. Skipping.')
                return None
            pil_img = crop_and_resize(pil_img, size_bucket)
            pil_imgs.append(pil_img)

        latents = encode_pil_to_latents(pil_imgs, vae)
        return {'latents': latents}

    return fn


def is_dev(safetensors_path):
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            if key.startswith('guidance_in'):
                return True
    return False


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


class CustomFluxPipeline:
    # Unique name, used to make the cache_dir path.
    name = 'flux'

    # layers that will participate in activation checkpointing
    checkpointable_layers = [
        'TransformerWrapper',
        'SingleTransformerWrapper',
    ]

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        kwargs = {}
        if transformer_path := self.model_config.get('transformer', None):
            transformer_config = 'configs/flux_dev_config.json' if is_dev(transformer_path) else 'configs/flux_schnell_config.json'
            transformer = diffusers.FluxTransformer2DModel.from_single_file(
                self.model_config['transformer'],
                torch_dtype=self.model_config['dtype'],
                config=transformer_config,
                local_files_only=True,
            )
            kwargs['transformer'] = transformer
        self.diffusers_pipeline = diffusers.FluxPipeline.from_pretrained(self.model_config['diffusers_path'], torch_dtype=self.model_config['dtype'], **kwargs)
        self.transformer.train()

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_modules(self):
        return self.vae, self.text_encoder, self.text_encoder_2

    def configure_adapter(self, adapter_config):
        target_linear_modules = []
        for module in self.transformer.modules():
            if module.__class__.__name__ not in ADAPTER_TARGET_MODULES:
                continue
            for name, submodule in module.named_modules():
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.append(name)

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        lora_model = peft.get_peft_model(self.transformer, peft_config)
        #self.transformer.add_adapter(peft_config)
        if is_main_process():
            lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])
        return peft_config

    def save_adapter(self, save_dir, peft_state_dict):
        adapter_type = self.config['adapter']['type']
        if adapter_type == 'lora':
            self.save_lora_weights(save_dir, transformer_lora_layers=peft_state_dict)
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    def get_dataset_map_fn(self, module, size_bucket):
        if module == self.vae:
            return process_image_fn(module, size_bucket)
        elif module == self.text_encoder:
            def fn(example):
                return {'clip_embed': self._get_clip_prompt_embeds(prompt=example['caption'], device=module.device).to('cpu')}
            return fn
        elif module == self.text_encoder_2:
            def fn(example):
                return {'t5_embed': self._get_t5_prompt_embeds(prompt=example['caption'], device=module.device).to('cpu')}
            return fn
        else:
            raise RuntimeError(f'Module {module.__class__} does not have a map fn implemented')

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents']
        clip_embed = inputs['clip_embed']
        t5_embed = inputs['t5_embed']

        # The following code taken and slightly modified from x-flux (https://github.com/XLabs-AI/x-flux/tree/main)
        bs, c, h, w = latents.shape
        latents = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        img_ids = self._prepare_latent_image_ids(bs, h, w, latents.device, latents.dtype)
        if img_ids.ndim == 2:
            # This method must return tensors with batch dimension, since we proceed to split along batch dimension for pipelining.
            img_ids = img_ids.unsqueeze(0).repeat((bs, 1, 1))
        txt_ids = torch.zeros(bs, t5_embed.shape[1], 3).to(latents.device, latents.dtype)

        if timestep_quantile is not None:
            dist = torch.distributions.normal.Normal(0, 1)
            logits_norm = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            logits_norm = torch.randn((bs,), device=latents.device)

        sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
        logits_norm = logits_norm * sigmoid_scale
        t = torch.sigmoid(logits_norm)
        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1
        guidance_vec = torch.full((x_t.shape[0],), float(self.model_config['guidance']), device=x_t.device, dtype=torch.float32)

        model_dtype = self.model_config['dtype']
        features = (x_t.to(model_dtype), t5_embed.to(model_dtype), clip_embed.to(model_dtype), t, img_ids, txt_ids, guidance_vec, target)

        # We pass the target through the layers of the model in the features tuple, so that it matches the noisy input when we get to the
        # last pipeline parallel stage.
        return features

    def to_layers(self):
        transformer = self.transformer
        layers = [EmbeddingWrapper(transformer.x_embedder, transformer.time_text_embed, transformer.context_embedder, transformer.pos_embed)]
        for block in transformer.transformer_blocks:
            layers.append(TransformerWrapper(block))
        layers.append(concatenate_hidden_states)
        for block in transformer.single_transformer_blocks:
            layers.append(SingleTransformerWrapper(block))
        layers.append(OutputWrapper(transformer.norm_out, transformer.proj_out))
        return layers


class EmbeddingWrapper(nn.Module):
    def __init__(self, x_embedder, time_text_embed, context_embedder, pos_embed):
        super().__init__()
        self.x_embedder = x_embedder
        self.time_text_embed = time_text_embed
        self.context_embedder = context_embedder
        self.pos_embed = pos_embed

    def forward(self, inputs):
        # Don't know why I have to do this. I had to do it in qlora-pipe also.
        # Without it, you get RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        for item in inputs:
            item.requires_grad_(True)
        hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, target = inputs
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000
        # handle dev vs schnell
        if self.time_text_embed.__class__.__name__ == 'CombinedTimestepGuidanceTextProjEmbeddings':
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        freqs_cos, freqs_sin = self.pos_embed(ids)
        return hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target


class TransformerWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target = inputs
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(freqs_cos, freqs_sin),
        )
        return hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target


def concatenate_hidden_states(inputs):
    hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target = inputs
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    return hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target


class SingleTransformerWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target = inputs
        hidden_states = self.block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=(freqs_cos, freqs_sin),
        )
        return hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target


class OutputWrapper(nn.Module):
    def __init__(self, norm_out, proj_out):
        super().__init__()
        self.norm_out = norm_out
        self.proj_out = proj_out

    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, target = inputs
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        output = output.to(torch.float32)
        target = target.to(torch.float32)
        loss = F.mse_loss(output, target)
        return loss
