import argparse
import os
from datetime import datetime, timezone
import shutil
import glob

import toml
import deepspeed
from deepspeed.runtime.pipe import module as ds_pipe_module
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils import dataset as dataset_util
from utils.common import is_main_process, DTYPE_MAP, empty_cuda_cache
import utils.saver
from models import flux

CHECKPOINTABLE_LAYERS = [
    'TransformerWrapper',
    'SingleTransformerWrapper',
]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None, help='resume training from the most recent checkpoint')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('save_every_n_epochs', 1)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    model_config.setdefault('guidance', 1.0)

    if 'lora' in config:
        lora_config = config['lora']
        lora_config.setdefault('alpha', lora_config['rank'])
        lora_config.setdefault('dropout', 0.0)
        lora_config.setdefault('dtype', model_dtype_str)
        lora_config['dtype'] = DTYPE_MAP[lora_config['dtype']]

    dataset_config = config['dataset']
    dataset_config.setdefault('shuffle_tags', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


if __name__ == '__main__':
    with open(args.config) as f:
        config = toml.load(f)
    set_config_defaults(config)

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )

    deepspeed.init_distributed()

    model_config = config['model']
    model_type = model_config['type']

    if model_type == 'flux':
        model = flux.CustomFluxPipeline.from_pretrained(model_config['path'], torch_dtype=model_config['dtype'])
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')
    model.model_config = model_config
    model.transformer.train()

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    lora_model, peft_config = model.inject_lora_layers(config['lora'])
    for name, p in lora_model.named_parameters():
        p.original_name = name
        if p.requires_grad:
            p.data = p.data.to(config['lora']['dtype'])
    if is_main_process():
        lora_model.print_trainable_parameters()

    batch_size = config.get('batch_size', 1)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    train_data = dataset_util.Dataset(config['dataset'], model, data_parallel_rank=0, data_parallel_world_size=1, batch_size=batch_size*gradient_accumulation_steps)

    # if this is a new run, create a new dir for it
    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(args.deepspeed_config, run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    deepspeed.comm.barrier()
    run_dir = get_most_recent_run_dir(config['output_dir'])

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    if config['activation_checkpointing']:
        checkpoint_func = deepspeed.checkpointing.checkpoint
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': CHECKPOINTABLE_LAYERS,
            'activation_checkpoint_func': checkpoint_func,
        })
    pipeline_model = deepspeed.pipe.PipelineModule(
        layers=layers,
        num_stages=config['pipeline_stages'],
        partition_method='parameters',
        loss_fn=model.get_loss_fn(),
        **additional_pipeline_module_kwargs
    )
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    def get_optimizer(model_parameters):
        optim_config = config['optimizer']
        lr = optim_config['lr']
        optim_type = optim_config['type'].lower()
        if optim_type == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            return deepspeed.ops.adam.FusedAdam(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
                weight_decay=optim_config.get('weight_decay', 0.01),
                eps=optim_config.get('eps', 1e-6)
            )
        elif optim_type == 'adamw8bit':
            return bitsandbytes.optim.AdamW8bit(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
                weight_decay=optim_config.get('weight_decay', 0.01),
                eps=optim_config.get('eps', 1e-6)
            )
        elif optim_type == 'adamw_kahan':
            import optimi
            return optimi.AdamW(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
                weight_decay=optim_config.get('weight_decay', 0.01),
                kahan_sum=optim_config.get('kahan_sum', True),
                eps=optim_config.get('eps', 1e-6)
            )
        else:
            raise NotImplementedError(optim_type)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
    )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, gradient_accumulation_steps)
    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    step = 1
    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, peft_config, run_dir, model, train_dataloader, model_engine, pipeline_model)

    while True:
        #empty_cuda_cache()
        loss = model_engine.train_batch()
        model_engine.reset_activation_shape()
        train_dataloader.sync_epoch()

        epoch = saver.process_epoch(epoch, step)
        if epoch is None:
            break

        step += 1

    if is_main_process():
        print('TRAINING COMPLETE!')