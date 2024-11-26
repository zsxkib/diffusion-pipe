from pathlib import Path
import os.path
import random
from collections import defaultdict
import math

import torch
from deepspeed.utils.logging import logger
from deepspeed import comm as dist
import datasets
from datasets.fingerprint import Hasher
from PIL import Image
import imageio
import numpy as np

from utils.common import zero_first, empty_cuda_cache, is_main_process, VIDEO_EXTENSIONS


DEBUG = False
IMAGE_SIZE_ROUND_TO_MULTIPLE = 32


def shuffle_with_seed(l, seed=None):
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(l)
    random.setstate(rng_state)


def process_caption_fn(shuffle_tags=False, caption_prefix=''):
    def fn(example):
        with open(example['caption_file']) as f:
            caption = f.read().strip()
        if shuffle_tags:
            tags = [tag.strip() for tag in caption.split(',')]
            random.shuffle(tags)
            caption = ', '.join(tags)
        caption = caption_prefix + caption

        example['caption'] = caption
        return example
    return fn


def round_to_multiple(x, multiple):
    return int((x // multiple) * multiple)


def _map_and_cache(dataset, map_fn, cache_dir, cache_file_prefix='', new_fingerprint_args=None, regenerate_cache=False, caching_batch_size=1):
    # Do the fingerprinting ourselves, because otherwise map() does it by serializing the map function.
    # That goes poorly when the function is capturing huge models (slow, OOMs, etc).
    new_fingerprint_args = [] if new_fingerprint_args is None else new_fingerprint_args
    new_fingerprint_args.append(dataset._fingerprint)
    new_fingerprint = Hasher.hash(new_fingerprint_args)
    cache_file = cache_dir / f'{cache_file_prefix}{new_fingerprint}.arrow'
    if (not is_main_process()) or (cache_file.exists() and not regenerate_cache):
        logger.info('Dataset fingerprint matched cache, loading from cache')
        assert cache_file.exists()
    else:
        logger.info('Dataset fingerprint changed, removing existing cache file and regenerating')
        for existing_cache_file in cache_dir.glob(f'{cache_file_prefix}*.arrow'):
            existing_cache_file.unlink()
    # lower writer_batch_size from the default of 1000 or we get a weird pyarrow overflow error
    dataset = dataset.map(
        map_fn,
        cache_file_name=str(cache_file),
        writer_batch_size=100,
        new_fingerprint=new_fingerprint,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=caching_batch_size,
    )
    return dataset


# The smallest unit of a dataset. Represents a single size bucket from a single folder of images
# and captions on disk. Not batched; returns individual items.
class SizeBucketDataset:
    def __init__(self, image_file_and_caption_dataset, directory_config, size_bucket, model, regenerate_cache=False, caching_batch_size=1):
        self.image_file_and_caption_dataset = image_file_and_caption_dataset
        self.directory_config = directory_config
        self.size_bucket = size_bucket
        self.model = model
        self.regenerate_cache = regenerate_cache
        self.caching_batch_size = caching_batch_size
        self.path = Path(self.directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model.name / f'cache_{size_bucket[0]}x{size_bucket[1]}x{size_bucket[2]}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.datasets = []
        self.num_repeats = self.directory_config.get('num_repeats', 1)
        if is_main_process():
            logger.info(f'size_bucket: {size_bucket}, num_images: {len(self.image_file_and_caption_dataset)}, num_repeats: {self.num_repeats}')

    def cache_latents(self, vae):
        self.datasets.append(
            _map_and_cache(
                self.image_file_and_caption_dataset,
                self.model.get_latents_map_fn(vae, self.size_bucket),
                self.cache_dir,
                cache_file_prefix='latents_',
                regenerate_cache=self.regenerate_cache,
                caching_batch_size=self.caching_batch_size
            )
        )

    def add_dataset(self, te_dataset):
        self.datasets.append(te_dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.image_file_and_caption_dataset)
        if DEBUG:
            print(Path(self.image_file_and_caption_dataset[idx]['image_file']).stem)
        ret = {}
        for ds in self.datasets:
            ret.update(ds[idx])
        return ret

    def __len__(self):
        return len(self.image_file_and_caption_dataset) * self.num_repeats


# Logical concatenation of multiple SizeBucketDataset, for the same size bucket. It returns items
# as batches.
class ConcatenatedBatchedDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.post_init_called = False
        iteration_order = []
        for i, ds in enumerate(self.datasets):
            iteration_order.extend([i]*len(ds))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.datasets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = iteration_order
        assert len(self.iteration_order) > 0, 'ConcatenatedBatchedDataset is empty. Are your file paths correct?'

    def post_init(self, batch_size):
        self.batch_size = batch_size
        self._make_divisible_by(self.batch_size)
        self.post_init_called = True

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order) // self.batch_size

    def __getitem__(self, idx):
        assert self.post_init_called
        start = idx * self.batch_size
        end = start + self.batch_size
        return [self.datasets[i][j] for i, j in self.iteration_order[start:end]]

    def _make_divisible_by(self, n):
        new_length = (len(self.iteration_order) // n) * n
        self.iteration_order = self.iteration_order[:new_length]
        if new_length == 0 and is_main_process():
            logger.warning(f"size bucket {self.datasets[0].size_bucket} is being completely dropped because it doesn't have enough images")


class ARBucketDataset:
    def __init__(self, ar_frames, resolutions, filepaths, directory_config, model, regenerate_cache=False, caching_batch_size=1):
        self.ar_frames = ar_frames
        self.resolutions = resolutions
        self.filepaths = filepaths
        self.directory_config = directory_config
        self.model = model
        self.size_buckets = []
        self.path = Path(directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model.name / f'ar_frames_{self.ar_frames[0]:.3f}_{self.ar_frames[1]}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.regenerate_cache = regenerate_cache
        self.caching_batch_size = caching_batch_size

        image_and_caption_files = self.filepaths
        # This is the one place we shuffle the data. Use a fixed seed, so the dataset is identical on all processes.
        # Processes other than rank 0 will then load it from cache.
        shuffle_with_seed(image_and_caption_files, seed=0)
        image_files, caption_files = zip(*image_and_caption_files)
        ds = datasets.Dataset.from_dict({'image_file': image_files, 'caption_file': caption_files})
        self.image_file_and_caption_dataset = ds.map(
            process_caption_fn(shuffle_tags=self.directory_config['shuffle_tags'], caption_prefix=self.directory_config['caption_prefix']),
            remove_columns='caption_file',
            keep_in_memory=True
        )
        self.image_file_and_caption_dataset.set_format('torch')

        for res in resolutions:
            area = res**2
            w = math.sqrt(area * self.ar_frames[0])
            h = area / w
            w = round_to_multiple(w, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            h = round_to_multiple(h, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            size_bucket = (w, h, self.ar_frames[1])
            self.size_buckets.append(
                SizeBucketDataset(self.image_file_and_caption_dataset, directory_config, size_bucket, model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)
            )

    def get_size_bucket_datasets(self):
        return self.size_buckets

    def cache_latents(self, vae):
        for ds in self.size_buckets:
            ds.cache_latents(vae)

    def cache_text_embeddings(self, text_encoder, i):
        te_dataset = _map_and_cache(
            self.image_file_and_caption_dataset,
            self.model.get_text_embeddings_map_fn(text_encoder),
            self.cache_dir,
            cache_file_prefix=f'text_embeddings_{i}_',
            new_fingerprint_args=[i],
            caching_batch_size=self.caching_batch_size,
            regenerate_cache=self.regenerate_cache
        )
        for size_bucket_dataset in self.size_buckets:
            size_bucket_dataset.add_dataset(te_dataset)


class DirectoryDataset:
    def __init__(self, directory_config, dataset_config, model, regenerate_cache=False, caching_batch_size=1):
        self._set_defaults(directory_config, dataset_config)
        self.directory_config = directory_config
        self.dataset_config = dataset_config
        self.enable_ar_bucket = directory_config.get('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        self.resolutions = directory_config.get('resolutions', dataset_config['resolutions'])

        if not self.enable_ar_bucket:
            ars = np.array([1.0])
        else:
            min_ar = self.directory_config.get('min_ar', self.dataset_config['min_ar'])
            max_ar = self.directory_config.get('max_ar', self.dataset_config['max_ar'])
            num_ar_buckets = self.directory_config.get('num_ar_buckets', self.dataset_config['num_ar_buckets'])
            ars = np.geomspace(min_ar, max_ar, num=num_ar_buckets)
        log_ars = np.log(ars)
        frame_buckets = self.directory_config.get('frame_buckets', self.dataset_config.get('frame_buckets', [1]))
        frame_buckets = np.array(frame_buckets)

        ar_bucket_to_filepaths = defaultdict(list)
        path = Path(self.directory_config['path'])
        if not path.exists() or not path.is_dir():
            raise RuntimeError(f'Invalid path: {path}')
        files = list(Path(path).glob('*'))
        # deterministic order
        files.sort()
        for file in files:
            if not file.is_file() or file.suffix == '.txt' or file.suffix == '.npz':
                continue
            image_file = file
            caption_file = image_file.with_suffix('.txt')
            if not os.path.exists(caption_file):
                if is_main_process():
                    logger.warning(f'Image file {image_file} does not have corresponding caption file. Skipping.')
                continue
            ar_bucket = self._find_closest_ar_bucket(image_file, ars, log_ars, frame_buckets)
            if ar_bucket is not None:
                ar_bucket_to_filepaths[ar_bucket].append((str(image_file), str(caption_file)))

        self.ar_buckets = []
        for ar_bucket, filepaths in ar_bucket_to_filepaths.items():
            self.ar_buckets.append(
                ARBucketDataset(
                    ar_bucket,
                    self.resolutions,
                    filepaths,
                    directory_config,
                    model,
                    regenerate_cache=regenerate_cache,
                    caching_batch_size=caching_batch_size,
                )
            )

    def _set_defaults(self, directory_config, dataset_config):
        directory_config.setdefault('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        directory_config.setdefault('resolutions', dataset_config['resolutions'])
        directory_config.setdefault('shuffle_tags', dataset_config.get('shuffle_tags', False))
        directory_config.setdefault('caption_prefix', dataset_config.get('caption_prefix', ''))

    def _find_closest_ar_bucket(self, image_file, ars, log_ars, frame_buckets):
        try:
            if image_file.suffix in VIDEO_EXTENSIONS:
                img = imageio.v3.imread(image_file)
                frames, height, width = img.shape[-4:-1]
            else:
                pil_img = Image.open(image_file)
                width, height = pil_img.size
                frames = 1
        except Exception:
            if is_main_process():
                logger.warning(f'Image file {image_file} could not be opened. Skipping.')
            return None
        log_ar = np.log(width / height)
        # Best AR bucket is the one with the smallest AR difference in log space.
        i = np.argmin(np.abs(log_ar - log_ars))
        j = np.argmin(np.abs(frames - frame_buckets))
        return (ars[i], frame_buckets[j])

    def get_size_bucket_datasets(self):
        result = []
        for ar_bucket_dataset in self.ar_buckets:
            result.extend(ar_bucket_dataset.get_size_bucket_datasets())
        return result

    def cache_latents(self, vae):
        for ds in self.ar_buckets:
            ds.cache_latents(vae)

    def cache_text_embeddings(self, text_encoder, i):
        for ds in self.ar_buckets:
            ds.cache_text_embeddings(text_encoder, i)


# Outermost dataset object that the caller uses. Contains multiple ConcatenatedBatchedDataset. Responsible
# for returning the correct batch for the process's data parallel rank. Calls model.prepare_inputs so the
# returned tuple of tensors is whatever the model needs.
class Dataset:
    def __init__(self, dataset_config, model, regenerate_cache=False, caching_batch_size=1):
        super().__init__()
        self.dataset_config = dataset_config
        self.model = model
        self.post_init_called = False
        self.eval_quantile = None

        self.directory_datasets = []
        datasets_by_size_bucket = defaultdict(list)
        for directory_config in dataset_config['directory']:
            directory_dataset = DirectoryDataset(directory_config, dataset_config, model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)
            self.directory_datasets.append(directory_dataset)
            for size_bucket_dataset in directory_dataset.get_size_bucket_datasets():
                datasets_by_size_bucket[size_bucket_dataset.size_bucket].append(size_bucket_dataset)

        self.buckets = []
        for datasets in datasets_by_size_bucket.values():
            self.buckets.append(ConcatenatedBatchedDataset(datasets))

    def post_init(self, data_parallel_rank, data_parallel_world_size, per_device_batch_size, gradient_accumulation_steps):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.batch_size = per_device_batch_size * gradient_accumulation_steps
        self.global_batch_size = self.data_parallel_world_size * self.batch_size

        for bucket in self.buckets:
            bucket.post_init(self.global_batch_size)

        iteration_order = []
        for i, bucket in enumerate(self.buckets):
            iteration_order.extend([i]*(len(bucket)))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.buckets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = iteration_order
        if DEBUG:
            print(f'Dataset iteration_order: {self.iteration_order}')

        self.post_init_called = True

        if subsample_ratio := self.dataset_config.get('subsample_ratio', None):
            if is_main_process():
                logger.info(f'Subsampling dataset with ratio {subsample_ratio}')
            new_len = int(len(self) * subsample_ratio)
            self.iteration_order = self.iteration_order[:new_len]

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order)

    def __getitem__(self, idx):
        assert self.post_init_called
        i, j = self.iteration_order[idx]
        examples = self.buckets[i][j]
        start_idx = self.data_parallel_rank*self.batch_size
        examples_for_this_dp_rank = examples[start_idx:start_idx+self.batch_size]
        if DEBUG:
            print((start_idx, start_idx+self.batch_size))
        batch =  self._collate(examples_for_this_dp_rank)
        return self.model.prepare_inputs(batch, timestep_quantile=self.eval_quantile)

    # collates a list of dictionaries of tensors into a single dictionary of batched tensors
    def _collate(self, examples):
        ret = {}
        for key in examples[0].keys():
            ret[key] = torch.stack([example[key] for example in examples])
        return ret

    def cache_latents(self, vae):
        for ds in self.directory_datasets:
            ds.cache_latents(vae)

    def cache_text_embeddings(self, text_encoder, i):
        for ds in self.directory_datasets:
            ds.cache_text_embeddings(text_encoder, i)


# Helper class to make caching multiple datasets more efficient by moving
# models to GPU as few times as needed.
class DatasetManager:
    def __init__(self, model):
        self.model = model
        self.datasets = []

    def register(self, dataset):
        self.datasets.append(dataset)

    def cache(self):
        with zero_first():
            self._cache()

    @torch.no_grad()
    def _cache(self):
        vae = self.model.get_vae()
        if is_main_process():
            vae.to('cuda')
            logger.info('Caching latents')
        for ds in self.datasets:
            ds.cache_latents(vae)
        vae.to('cpu')
        empty_cuda_cache()

        for i, text_encoder in enumerate(self.model.get_text_encoders()):
            if is_main_process():
                text_encoder.to('cuda')
                logger.info(f'Caching text embeddings {i}')
            for ds in self.datasets:
                ds.cache_text_embeddings(text_encoder, i)
            text_encoder.to('cpu')
            empty_cuda_cache()


def split_batch(batch, pieces):
    example_tuple = batch
    split_size = example_tuple[0].size(0) // pieces
    split_examples = zip(*(torch.split(tensor, split_size) for tensor in example_tuple))
    # Deepspeed works with a tuple of (features, labels), even if we don't provide a loss_fn to PipelineEngine,
    # and instead compute the loss ourselves in the model. It's okay to just return None for the labels here.
    return [(ex, None) for ex in split_examples]


# DataLoader that divides batches into microbatches for gradient accumulation steps when doing
# pipeline parallel training. Iterates indefinitely (deepspeed requirement). Keeps track of epoch.
# Updates epoch as soon as the final batch is returned (notably different from qlora-pipe).
class PipelineDataLoader:
    def __init__(self, dataset, gradient_accumulation_steps):
        self.dataset = dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.skip_first_n_batches = None
        self.iter_called = False
        self.reset()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None

    def __iter__(self):
        self.iter_called = True
        self._create_dataloader()
        return self

    def __len__(self):
        return len(self.dataset) * self.gradient_accumulation_steps

    def __next__(self):
        if self.next_micro_batch == None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            assert self.skip_first_n_batches is None
            self._create_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = next(self.data)
            self.epoch += 1
        return ret

    def _create_dataloader(self):
        if self.skip_first_n_batches is not None:
            sampler = SkipFirstNSampler(self.skip_first_n_batches, len(self.dataset))
            self.skip_first_n_batches = None
        else:
            sampler = None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=True,
            batch_size=None,
            sampler=sampler,
        )
        self.data = self._pull_batches_from_dataloader()

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

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        assert not self.iter_called
        self.epoch = state_dict['epoch']
        # -1 because by preloading the next micro_batch, it's always going to have one more batch
        # pulled than the actual number of batches iterated by the caller.
        self.num_batches_pulled = state_dict['num_batches_pulled'] - 1
        self.skip_first_n_batches = self.num_batches_pulled


class SkipFirstNSampler(torch.utils.data.Sampler):
    def __init__(self, n, dataset_length):
        super().__init__()
        self.n = n
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __iter__(self):
        for i in range(self.n, self.dataset_length):
            yield i


if __name__ == '__main__':
    from utils import common
    common.is_main_process = lambda: True
    from contextlib import contextmanager
    @contextmanager
    def _zero_first():
        yield
    common.zero_first = _zero_first

    from utils import dataset as dataset_util
    dataset_util.DEBUG = True

    from models import flux
    model = flux.CustomFluxPipeline.from_pretrained('/data2/imagegen_models/FLUX.1-dev', torch_dtype=torch.bfloat16)
    model.model_config = {'guidance': 1.0, 'dtype': torch.bfloat16}

    import toml
    dataset_manager = dataset_util.DatasetManager(model)
    with open('/home/anon/code/diffusion-pipe-configs/datasets/tiny1.toml') as f:
        dataset_config = toml.load(f)
    train_data = dataset_util.Dataset(dataset_config, model)
    dataset_manager.register(train_data)
    dataset_manager.cache()

    train_data.post_init(data_parallel_rank=0, data_parallel_world_size=1, per_device_batch_size=1, gradient_accumulation_steps=2)
    print(f'Dataset length: {len(train_data)}')

    for item in train_data:
        pass