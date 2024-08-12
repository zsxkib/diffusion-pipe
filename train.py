import argparse

import toml
import deepspeed

from utils import dataset
from utils import common
from models import flux

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=None, help='resume training from the most recent checkpoint')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


def set_config_defaults(config):
    config['model']['dtype'] = common.DTYPE_MAP[config['model']['dtype']]


if __name__ == '__main__':
    with open(args.config) as f:
        config = toml.load(f)
    set_config_defaults(config)

    deepspeed.init_distributed()

    model_config = config['model']
    model_type = model_config['type']

    if model_type == 'flux':
        model = flux.CustomFluxPipeline.from_pretrained(model_config['path'], torch_dtype=model_config['dtype'])
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

    batch_size = config.get('batch_size', 1)
    dataset = dataset.Dataset(config['dataset'], model, data_parallel_rank=0, data_parallel_world_size=1, batch_size=2)

    for item in dataset:
        print(item['latents'].size())
