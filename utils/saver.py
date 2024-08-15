from pathlib import Path
import os
import shutil

import torch
from deepspeed import comm as dist
from deepspeed.utils.logging import logger
import safetensors

from utils.common import is_main_process


def convert_state_dict_dtype(state_dict, dtype):
    for key, v in state_dict.items():
        state_dict[key] = v.to(device='cpu', dtype=dtype)


class Saver:
    def __init__(self, args, config, peft_config, save_root, train_dataloader, model_engine, pipeline_model):
        self.args = args
        self.config = config
        self.peft_config = peft_config
        self.save_root = Path(save_root)
        self.train_dataloader = train_dataloader
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model

    def save_lora(self, name):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / 'tmp'
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()
        if dp_id == 0:
            partial_state_dict = {}
            for name, p in self.pipeline_model.named_parameters():
                if p.requires_grad:
                    if not hasattr(p, 'original_name'):
                        logger.warning(f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.')
                        continue
                    partial_state_dict[p.original_name.replace('.default', '').replace('.modules_to_save', '')] = p.detach()
                    if 'save_dtype' in self.config:
                        convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
            torch.save(partial_state_dict, tmp_dir / f'state_dict_{stage_id}.bin')
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in tmp_dir.glob('*.bin'):
                state_dict.update(torch.load(path, weights_only=True, map_location='cpu'))
            safetensors.torch.save_file(state_dict, save_dir / 'adapter_model.safetensors')
            self.peft_config.save_pretrained(save_dir)
            shutil.copy(self.args.config, save_dir)
            shutil.copy(self.args.deepspeed_config, save_dir)
            shutil.rmtree(tmp_dir)

    def save_full_model(self, name):
        raise NotImplementedError()

    def save_model(self, name):
        if self.peft_config is not None:
            self.save_lora(name)
        else:
            self.save_full_model(name)

    def process_epoch(self, epoch, step):
        if self.train_dataloader.epoch != epoch:
            #self.save_checkpoint(step)
            if epoch % self.config['save_every_n_epochs'] == 0:
                if is_main_process():
                    print('Saving model')
                self.save_model(f'epoch{epoch}')
            epoch = self.train_dataloader.epoch
            if epoch > self.config['epochs']:
                return None
            if is_main_process():
                print(f'Started new epoch: {epoch}')
        return epoch