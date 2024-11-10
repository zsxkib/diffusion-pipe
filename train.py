import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time
import random

import toml
import deepspeed
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import dataset as dataset_util
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.isolate_rng import isolate_rng
from models import flux

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None, help='resume training from the most recent checkpoint')
parser.add_argument('--regenerate_cache', action='store_true', default=None, help='Force regenerate cache. Useful if none of the files have changed but their contents have, e.g. modified captions.')
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
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            adapter_config.setdefault('alpha', adapter_config['rank'])
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_before_first_step', True)


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


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.dataset.set_eval_quantile(quantile)
    orig_micro_batches = model_engine.micro_batches
    model_engine.micro_batches = eval_gradient_accumulation_steps
    iterator = iter(eval_dataloader)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        loss = model_engine.eval_batch(iterator).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    model_engine.micro_batches = orig_micro_batches
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        pbar.close()


def evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    if len(eval_dataloaders) == 0:
        return
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)


if __name__ == '__main__':
    with open(args.config) as f:
        config = toml.load(f)
    set_config_defaults(config)

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    deepspeed.init_distributed()

    model_type = config['model']['type']

    if model_type == 'flux':
        model = flux.CustomFluxPipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    if 'adapter' in config:
        peft_config = model.configure_adapter(config['adapter'])
    else:
        peft_config = None

    dataset_manager = dataset_util.DatasetManager(model)
    with open(config['dataset']) as f:
        dataset_config = toml.load(f)
    caching_batch_size = config.get('caching_batch_size', 1)
    train_data = dataset_util.Dataset(dataset_config, model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)
        dataset_manager.register(eval_data_map[name])

    dataset_manager.cache()

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
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })
    pipeline_model = deepspeed.pipe.PipelineModule(
        layers=layers,
        num_stages=config['pipeline_stages'],
        partition_method=config.get('partition_method', 'parameters'),
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
            # return deepspeed.ops.adam.FusedAdam(
            #     model_parameters,
            #     lr=lr,
            #     betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
            #     weight_decay=optim_config.get('weight_decay', 0.01),
            #     eps=optim_config.get('eps', 1e-8)
            # )
            return torch.optim.AdamW(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
                weight_decay=optim_config.get('weight_decay', 0.01),
                eps=optim_config.get('eps', 1e-8)
            )
        elif optim_type == 'adamw8bit':
            return bitsandbytes.optim.AdamW8bit(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
                weight_decay=optim_config.get('weight_decay', 0.01),
                eps=optim_config.get('eps', 1e-8)
            )
        elif optim_type == 'adamw_kahan':
            import optimi
            return optimi.AdamW(
                model_parameters,
                lr=lr,
                betas=(optim_config.get('beta1', 0.9), optim_config.get('beta2', 0.99)),
                weight_decay=optim_config.get('weight_decay', 0.01),
                kahan_sum=optim_config.get('kahan_sum', True),
                eps=optim_config.get('eps', 1e-8)
            )
        else:
            raise NotImplementedError(optim_type)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
    )

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            model_engine.train_micro_batch_size_per_gpu(),
            config['eval_gradient_accumulation_steps'],
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine.gradient_accumulation_steps())

    step = 1
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config,
        )
        deepspeed.comm.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(eval_data, config['eval_gradient_accumulation_steps'])
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, peft_config, run_dir, model, train_dataloader, model_engine, pipeline_model)

    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'])

    while True:
        #empty_cuda_cache()
        model_engine.reset_activation_shape()
        loss = model_engine.train_batch()
        train_dataloader.sync_epoch()

        new_epoch = saver.process_epoch(epoch, step)
        finished_epoch = True if new_epoch != epoch else False

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss.item(), step)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model_engine, eval_dataloaders, tb_writer, step, config['eval_gradient_accumulation_steps'])

        if finished_epoch:
            epoch = new_epoch
            if epoch is None:
                break
        step += 1

    if is_main_process():
        print('TRAINING COMPLETE!')