import diffusers
import peft
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from utils.common import is_main_process


class CustomFluxPipeline(diffusers.FluxPipeline):
    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return self.text_encoder, self.text_encoder_2

    def get_text_embedding(self, i, prompt):
        if i == 0:
            return self._get_clip_prompt_embeds(prompt=prompt, device=self.text_encoder.device)
        elif i == 1:
            return self._get_t5_prompt_embeds(prompt=prompt, device=self.text_encoder_2.device)
        else:
            raise ValueError(f'This model does not have a text encoder for index {i}')

    def inject_lora_layers(self, lora_config):
        # TODO: I yoinked this list from SimpleTuner. Need to read the flux code and make sure I
        # agree this is correct and covers all Linear layers.
        target_modules = [
            'to_k',
            'to_q',
            'to_v',
            'add_k_proj',
            'add_q_proj',
            'add_v_proj',
            'to_out.0',
            'to_add_out.0',
            'ff.0',
            'ff.2',
            'ff_context.0',
            'ff_context.2',
            'proj_mlp',
            'proj_out',
        ]
        peft_config = peft.LoraConfig(
            r=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout'],
            bias='none',
            target_modules=target_modules
        )
        lora_model = peft.get_peft_model(self.transformer, peft_config)
        return lora_model, peft_config

    def save_lora(self, save_dir, peft_state_dict):
        self.save_lora_weights(save_dir, transformer_lora_layers=peft_state_dict)

    def prepare_inputs(self, batch):
        img = batch['latents']

        # The following code taken and slightly modified from x-flux (https://github.com/XLabs-AI/x-flux/tree/main)
        bs, c, h, w = img.shape

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        # need .contiguous() for dataloader memory pinning
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs).contiguous()

        clip_embed = batch['text_embedding_1']
        t5_embed = batch['text_embedding_2']
        txt_ids = torch.zeros(bs, t5_embed.shape[1], 3)

        x_1 = img
        t = torch.sigmoid(torch.randn((bs,)))
        x_0 = torch.randn_like(x_1).to(x_1.device)
        x_t = (1 - t) * x_1 + t * x_0
        target = x_0 - x_1
        guidance_vec = torch.full((x_t.shape[0],), self.model_config['guidance'], device=x_t.device, dtype=x_t.dtype)

        model_dtype = self.model_config['dtype']
        features = (img.to(model_dtype), t5_embed.to(model_dtype), clip_embed.to(model_dtype), t, img_ids, txt_ids, guidance_vec.to(model_dtype))
        return (features, target)

    def get_loss_fn(self):
        def loss_fn(output, target):
            output = output.to(torch.float32)
            target = target.to(torch.float32)
            return F.mse_loss(output, target)
        return loss_fn

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
        hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance = inputs
        hidden_states = self.x_embedder(hidden_states)
        # Note: we only do the code path where guidance != None. Not sure why the other path even exists, we always have a guidance vector.
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        txt_ids = txt_ids.expand(img_ids.size(0), -1, -1)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)
        return hidden_states, encoder_hidden_states, temb, image_rotary_emb


class TransformerWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, image_rotary_emb = inputs
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        return hidden_states, encoder_hidden_states, temb, image_rotary_emb


def concatenate_hidden_states(inputs):
    hidden_states, encoder_hidden_states, temb, image_rotary_emb = inputs
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    return hidden_states, encoder_hidden_states, temb, image_rotary_emb


class SingleTransformerWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, image_rotary_emb = inputs
        hidden_states = self.block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        return hidden_states, encoder_hidden_states, temb, image_rotary_emb


class OutputWrapper(nn.Module):
    def __init__(self, norm_out, proj_out):
        super().__init__()
        self.norm_out = norm_out
        self.proj_out = proj_out

    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, image_rotary_emb = inputs
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return output
