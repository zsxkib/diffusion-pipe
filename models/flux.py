import diffusers
import peft
import torch
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
        if is_main_process():
            lora_model.print_trainable_parameters()

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
        t = torch.sigmoid(torch.randn((bs,))).view(-1, 1, 1)
        x_0 = torch.randn_like(x_1).to(x_1.device)
        x_t = (1 - t) * x_1 + t * x_0
        noise = x_0 - x_1
        guidance_vec = torch.full((x_t.shape[0],), self.model_config['guidance'], device=x_t.device, dtype=x_t.dtype)

        features = (img, t5_embed, clip_embed, t, img_ids, txt_ids, guidance_vec)
        label = noise
        return (features, label)
