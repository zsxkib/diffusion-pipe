import math
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules'))

import diffusers
import transformers
import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, load_state_dict

from Lumina_2.models.model import NextDiT_2B_GQA_patch2_Adaln_Refiner


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    return t


class Lumina2Pipeline(BasePipeline):
    name = 'lumina_2'
    checkpointable_layers = ['InitialLayer', 'TransformerLayer']
    # This will also train the noise_refiner and context_refiner layers, which aren't part of the main stack of transformer
    # layers, since they also use this class.
    adapter_target_modules = ['JointTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        dtype = self.model_config['dtype']

        self.vae = diffusers.AutoencoderKL.from_single_file(self.model_config['vae_path'], config='configs/flux_vae')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('configs/gemma_2_2b')
        self.tokenizer.padding_side = 'right'

        text_encoder_config = transformers.AutoConfig.from_pretrained('configs/gemma_2_2b')
        with init_empty_weights():
            self.text_encoder = transformers.AutoModel.from_config(text_encoder_config)
        state_dict = load_state_dict(self.model_config['llm_path'])
        for name, param in self.text_encoder.named_parameters():
            set_module_tensor_to_device(self.text_encoder, name, device='cpu', dtype=dtype, value=state_dict['model.'+name])

        self.text_encoder.eval()
        cap_feat_dim = self.text_encoder.config.hidden_size

        with init_empty_weights():
            self.transformer = NextDiT_2B_GQA_patch2_Adaln_Refiner(
                in_channels=16,
                qk_norm=True,
                cap_feat_dim=cap_feat_dim,
            )
        state_dict = load_state_dict(self.model_config['transformer_path'])
        for name, param in self.transformer.named_parameters():
            set_module_tensor_to_device(self.transformer, name, device='cpu', dtype=dtype, value=state_dict[name])

        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # args are lists
            assert not any(is_video)
            text_inputs = self.tokenizer(
                caption,
                padding='max_length',
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            prompt_masks = text_inputs.attention_mask

            device = self.text_encoder.device
            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids.to(device),
                attention_mask=prompt_masks.to(device),
                output_hidden_states=True,
            ).hidden_states[-2]
            return {'prompt_embeds': prompt_embeds, 'prompt_masks': prompt_masks}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        prompt_masks = inputs['prompt_masks']

        bs, c, h, w = latents.shape

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

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('lumina_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = latents - noise

        # If t is the amount of noise, then the timestep this model takes as input is 1-t.
        return noisy_latents, 1-t, prompt_embeds, prompt_masks, target

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.layers:
            layers.append(TransformerLayer(block))
        layers.append(FinalLayer(transformer))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t_embedder = model.t_embedder
        self.cap_embedder = model.cap_embedder
        self.rope_embedder = model.rope_embedder
        self.context_refiner = model.context_refiner
        self.x_embedder = model.x_embedder
        self.noise_refiner = model.noise_refiner
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, t, cap_feats, cap_mask, target = inputs

        t = self.t_embedder(t)
        adaln_input = t
        cap_feats = self.cap_embedder(cap_feats)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t)
        img_size = torch.tensor(img_size)
        cap_size = torch.tensor(cap_size)
        freqs_cis = freqs_cis.to(x.device)
        return make_contiguous(x, mask, freqs_cis, adaln_input, img_size, cap_size, target)


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, mask, freqs_cis, adaln_input, img_size, cap_size, target = inputs
        x = self.block(x, mask, freqs_cis, adaln_input)
        return make_contiguous(x, mask, freqs_cis, adaln_input, img_size, cap_size, target)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        # norm_final isn't used, but by registering it we will keep it in the saved model, preventing ComfyUI from logging a
        # warning that it's missing.
        self.norm_final = model.norm_final
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, mask, freqs_cis, adaln_input, img_size, cap_size, target = inputs
        x = self.final_layer(x, adaln_input)
        img_size = [(row[0].item(), row[1].item()) for row in img_size]
        cap_size = [row.item() for row in cap_size]
        output = self.unpatchify(x, img_size, cap_size, return_tensor=True)
        output = output.to(torch.float32)
        target = target.to(torch.float32)
        loss = F.mse_loss(output, target)
        return loss
