import peft
from torch import nn

from utils.common import is_main_process


class BasePipeline:
    def get_vae(self):
        raise NotImplementedError()

    def get_text_encoders(self):
        raise NotImplementedError()

    def configure_adapter(self, adapter_config):
        target_linear_modules = []
        for module in self.transformer.modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
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
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(self.transformer, peft_config)
        #self.transformer.add_adapter(peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])
        return peft_config

    def save_adapter(self, save_dir, peft_state_dict):
        raise NotImplementedError()

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_latents_map_fn(self, vae, size_bucket):
        raise NotImplementedError()

    def get_text_embeddings_map_fn(self, text_encoder):
        raise NotImplementedError()

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()
