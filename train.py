import argparse
import os
from datetime import datetime, timezone
import shutil
import glob

import toml
import deepspeed

from utils import dataset as dataset_util
from utils.common import is_main_process, DTYPE_MAP
from models import flux

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None, help='resume training from the most recent checkpoint')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


def set_config_defaults(config):
    model_config = config['model']
    model_config['dtype'] = DTYPE_MAP[model_config['dtype']]
    model_config.setdefault('guidance', 1.0)

    if 'lora' in config:
        lora_config = config['lora']
        lora_config.setdefault('alpha', lora_config['rank'])
        lora_config.setdefault('dropout', 0.0)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


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

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    model.inject_lora_layers(config['lora'])

    batch_size = config.get('batch_size', 1)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    train_data = dataset_util.Dataset(config['dataset'], model, data_parallel_rank=0, data_parallel_world_size=1, batch_size=batch_size*gradient_accumulation_steps)

    train_dataloader = dataset_util.PipelineDataLoader(train_data, gradient_accumulation_steps)
    item = next(train_dataloader)
    print(item[0][0].size())
    print(item[1].size())
    quit()

    # if this is a new run, create a new dir for it
    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(args.deepspeed_config, run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    deepspeed.comm.barrier()
    run_dir = get_most_recent_run_dir(config['output_dir'])
