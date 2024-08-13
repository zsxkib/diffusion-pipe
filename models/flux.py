import diffusers
import peft

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
