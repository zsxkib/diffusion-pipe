from pathlib import Path
import os.path
import random

import torch
from torchvision import transforms
from deepspeed.utils.logging import logger
import datasets
from PIL import Image, ImageOps
from datasets.fingerprint import Hasher

from . import common


def shuffle_with_seed(l, seed=None):
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(l)
    random.setstate(rng_state)


def crop_and_resize(pil_img):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    return ImageOps.fit(pil_img, (1024, 1024))


pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
def encode_pil_to_latents(pil_img, vae):
    img = pil_to_tensor(pil_img)
    img = img.unsqueeze(0)
    latents = vae.encode(img.to(vae.device, vae.dtype)).latent_dist.sample()
    if hasattr(vae.config, 'shift_factor'):
        latents = latents - vae.config.shift_factor
    latents = latents * vae.config.scaling_factor
    latents = latents.to('cpu')
    return latents


tensor_to_pil = transforms.Compose([transforms.Lambda(lambda x: (x / 2 + 0.5).clamp(0, 1)), transforms.ToPILImage()])
def decode_latents_to_pil(latents, vae):
    latents = latents.to(vae.device)
    latents = latents / vae.config.scaling_factor
    if hasattr(vae.config, 'shift_factor'):
        latents = latents + vae.config.shift_factor
    img = vae.decode(latents.to(vae.dtype), return_dict=False)[0].to(torch.float32)
    img = img.squeeze(0)
    return tensor_to_pil(img)


def process_caption_fn(shuffle_tags=False):
    def fn(example):
        with open(example['caption_file']) as f:
            caption = f.read().strip()
        if shuffle_tags:
            tags = [tag.strip() for tag in caption.split(',')]
            random.shuffle(tags)
            caption = ', '.join(tags)

        example['caption'] = caption
        return example
    return fn


def process_image_fn(vae):
    def fn(example):
        image_file = example['image_file']
        try:
            pil_img = Image.open(image_file)
        except Exception:
            logger.warning(f'Image file {image_file} could not be opened. Skipping.')
            return None
        pil_img = crop_and_resize(pil_img)
        latents = encode_pil_to_latents(pil_img, vae)

        example['latents'] = latents
        return example
    return fn


class Dataset:
    def __init__(self, dataset_config, batch_size, model):
        self.config = dataset_config
        self.batch_size = batch_size
        self.model = model
        self.path = Path(self.config['path'])
        self.cache_dir = self.path / 'cache'
        with common.zero_first():
            self._init()

    def _map_and_cache(self, dataset, map_fn, cache_file_prefix='', new_fingerprint_args=[]):
        # Do the fingerprinting ourselves, because otherwise map() does it by serializing the map function.
        # That goes poorly when the function is capturing huge models (slow, OOMs, etc).
        new_fingerprint_args.append(dataset._fingerprint)
        new_fingerprint = Hasher.hash(new_fingerprint_args)
        cache_file = self.cache_dir / f'{cache_file_prefix}{new_fingerprint}.arrow'
        if cache_file.exists():
            logger.info('Dataset fingerprint matched cache, loading from cache')
        else:
            logger.info('Dataset fingerprint changed, removing existing cache file and regenerating')
            for existing_cache_file in self.cache_dir.glob(f'{cache_file_prefix}*.arrow'):
                existing_cache_file.unlink()
        # lower writer_batch_size from the default of 1000 or we get a weird pyarrow overflow error
        dataset = dataset.map(map_fn, cache_file_name=str(cache_file), writer_batch_size=100, new_fingerprint=new_fingerprint)
        return dataset

    def _get_text_embedding_map_fn(self, i):
        def fn(example):
            example[f'text_embedding_{i}'] = self.model.get_text_embedding(i, example['caption'])
            return example
        return fn

    def _init(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        image_and_caption_files = []
        files = list(self.path.glob('*'))
        # deterministic order
        files.sort()
        for file in files:
            if not file.is_file() or file.suffix == '.txt':
                continue
            image_file = file
            caption_file = image_file.with_suffix('.txt')
            if not os.path.exists(caption_file):
                logger.warning(f'Image file {image_file} does not have corresponding caption file. Skipping.')
            image_and_caption_files.append((str(image_file), str(caption_file)))
        # This is the one place we shuffle the data. Use a fixed seed, so the dataset is identical on all processes.
        # Processes other than rank 0 will then load it from cache.
        shuffle_with_seed(image_and_caption_files, seed=0)
        image_files, caption_files = zip(*image_and_caption_files)
        dataset = datasets.Dataset.from_dict({'image_file': image_files, 'caption_file': caption_files})
        dataset = dataset.map(process_caption_fn(shuffle_tags=True), remove_columns='caption_file', keep_in_memory=True)
        dataset.set_format('torch')

        vae = self.model.get_vae()
        vae.to('cuda')
        with torch.no_grad():
            latent_dataset = self._map_and_cache(dataset, process_image_fn(vae), cache_file_prefix='latent_')
        vae.to('cpu')

        text_embedding_datasets = []
        for i, text_encoder in enumerate(self.model.get_text_encoders()):
            text_encoder.to('cuda')
            with torch.no_grad():
                text_embedding_datasets.append(self._map_and_cache(dataset, self._get_text_embedding_map_fn(i), cache_file_prefix=f'text_embedding_{i+1}_', new_fingerprint_args=[i]))
            text_encoder.to('cpu')

        print(latent_dataset[0])
        print(text_embedding_datasets[0][0])
        print(text_embedding_datasets[1][0])
