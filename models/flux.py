import diffusers


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
