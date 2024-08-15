from pathlib import Path
import os.path
import random

import torch
from torchvision import transforms
from deepspeed.utils.logging import logger
from deepspeed import comm as dist
import datasets
from PIL import Image, ImageOps
from datasets.fingerprint import Hasher

from utils.common import zero_first, empty_cuda_cache


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
    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
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

        example['latents'] = latents.squeeze(0)
        return example
    return fn


# Dataset that does caching, batching, and dividing batches across data parallel ranks.
# Logically represents a single folder of images and captions on disk.
class Dataset:
    def __init__(self, dataset_config, model, data_parallel_rank, data_parallel_world_size, batch_size):
        self.config = dataset_config
        self.model = model
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.batch_size = batch_size
        self.path = Path(self.config['path'])
        self.cache_dir = self.path / 'cache'
        with zero_first():
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
            example[f'text_embedding'] = self.model.get_text_embedding(i, example['caption']).to('cpu').squeeze(0)
            return example
        return fn

    def _make_divisible_by(self, n):
        self.length = self.num_examples // n
        self.num_examples = self.length * n

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
        dataset = dataset.map(process_caption_fn(shuffle_tags=self.config['shuffle_tags']), remove_columns='caption_file', keep_in_memory=True)
        dataset.set_format('torch')

        vae = self.model.get_vae()
        vae.to('cuda')
        with torch.no_grad():
            self.latent_dataset = self._map_and_cache(dataset, process_image_fn(vae), cache_file_prefix='latent_')
        vae.to('cpu')
        empty_cuda_cache()

        self.text_embedding_datasets = []
        for i, text_encoder in enumerate(self.model.get_text_encoders()):
            text_encoder.to('cuda')
            with torch.no_grad():
                self.text_embedding_datasets.append(self._map_and_cache(dataset, self._get_text_embedding_map_fn(i), cache_file_prefix=f'text_embedding_{i+1}_', new_fingerprint_args=[i]))
            text_encoder.to('cpu')
            empty_cuda_cache()
        self.num_examples = len(self.latent_dataset)
        for ds in self.text_embedding_datasets:
            assert len(ds) == self.num_examples
        self._make_divisible_by(self.data_parallel_world_size * self.batch_size)

    def __getitem__(self, i):
        if i >= len(self):
            return IndexError('Dataset index out of range')
        idx = (i * self.data_parallel_world_size + self.data_parallel_rank) * self.batch_size
        selector = slice(idx, idx + self.batch_size)
        d = {}
        d['latents'] = self.latent_dataset[selector]['latents']
        for te_idx, te_dataset in enumerate(self.text_embedding_datasets):
            d[f'text_embedding_{te_idx+1}'] = te_dataset[selector]['text_embedding']
        return self.model.prepare_inputs(d)

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def split_batch(batch, pieces):
    # batch is a tuple of (features, label) where features is also a tuple of tensors
    features, label = batch
    split_size = features[0].size(0) // pieces
    micro_batches = zip(zip(*(torch.split(tensor, split_size) for tensor in features)), torch.split(label, split_size))
    return micro_batches


# DataLoader that divides batches into microbatches for gradient accumulation steps when doing
# pipeline parallel training. Iterates indefinitely (deepspeed requirement). Keeps track of epoch.
class PipelineDataLoader:
    def __init__(self, dataset, gradient_accumulation_steps):
        self.dataset = dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.reset()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self._create_dataloader()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) * self.gradient_accumulation_steps

    def __next__(self):
        try:
            micro_batch = next(self.data)
        except StopIteration:
            self._create_dataloader()
            micro_batch = next(self.data)
            self.epoch += 1
        return micro_batch

    def _create_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=True,
            batch_size=None
        )
        self.data = self._pull_batches_from_dataloader()
        self.num_batches_pulled = 0

    def _pull_batches_from_dataloader(self):
        for batch in self.dataloader:
            self.num_batches_pulled += 1
            for micro_batch in split_batch(batch, self.gradient_accumulation_steps):
                yield micro_batch

    # Only the first and last stages in the pipeline pull from the dataloader. Parts of the code need
    # to know the epoch, so we synchronize the epoch so the processes that don't use the dataloader
    # know the current epoch.
    def sync_epoch(self):
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch
