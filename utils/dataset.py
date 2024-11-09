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

from utils.common import zero_first, empty_cuda_cache, is_main_process


DEBUG = False


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


# Modifed from: https://github.com/kohya-ss/sd-scripts/blob/main/library/model_util.py
def make_size_buckets(resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps):
    max_area = resolution**2

    resos = set()

    width = int(math.sqrt(max_area) // bucket_reso_steps) * bucket_reso_steps
    resos.add((width, width))

    width = min_bucket_reso
    while width <= max_bucket_reso:
        height = min(max_bucket_reso, int((max_area // width) // bucket_reso_steps) * bucket_reso_steps)
        if height >= min_bucket_reso:
            resos.add((width, height))
            resos.add((height, width))
        width += bucket_reso_steps

    resos = list(resos)
    resos.sort()
    return resos


# The smallest unit of a dataset. Represents a single size bucket from a single folder of images
# and captions on disk. Not batched; returns individual items.
class SizeBucketDataset:
    def __init__(self, filepaths, dataset_config, size_bucket, model, regenerate_cache=False, caching_batch_size=1):
        logger.info(f'size_bucket: {size_bucket}, num_images: {len(filepaths)}')
        self.filepaths = filepaths
        self.config = dataset_config
        self.config.setdefault('shuffle_tags', False)
        self.config.setdefault('caption_prefix', '')
        self.size_bucket = size_bucket
        self.model = model
        self.regenerate_cache = regenerate_cache
        self.caching_batch_size = caching_batch_size
        self.path = Path(self.config['path'])
        self.cache_dir = self.path / 'cache' / self.model.name / f'cache_{size_bucket[0]}x{size_bucket[1]}'
        self.datasets = []

        os.makedirs(self.cache_dir, exist_ok=True)
        image_and_caption_files = self.filepaths
        # This is the one place we shuffle the data. Use a fixed seed, so the dataset is identical on all processes.
        # Processes other than rank 0 will then load it from cache.
        shuffle_with_seed(image_and_caption_files, seed=0)
        image_files, caption_files = zip(*image_and_caption_files)
        ds = datasets.Dataset.from_dict({'image_file': image_files, 'caption_file': caption_files})
        self.image_file_and_caption_dataset = ds.map(process_caption_fn(shuffle_tags=self.config['shuffle_tags'], caption_prefix=self.config['caption_prefix']), remove_columns='caption_file', keep_in_memory=True)
        self.image_file_and_caption_dataset.set_format('torch')

    def _map_and_cache(self, dataset, map_fn, cache_file_prefix='', new_fingerprint_args=[]):
        # Do the fingerprinting ourselves, because otherwise map() does it by serializing the map function.
        # That goes poorly when the function is capturing huge models (slow, OOMs, etc).
        new_fingerprint_args.append(dataset._fingerprint)
        new_fingerprint = Hasher.hash(new_fingerprint_args)
        cache_file = self.cache_dir / f'{cache_file_prefix}{new_fingerprint}.arrow'
        if (not is_main_process()) or (cache_file.exists() and not self.regenerate_cache):
            logger.info('Dataset fingerprint matched cache, loading from cache')
        else:
            logger.info('Dataset fingerprint changed, removing existing cache file and regenerating')
            for existing_cache_file in self.cache_dir.glob(f'{cache_file_prefix}*.arrow'):
                existing_cache_file.unlink()
        # lower writer_batch_size from the default of 1000 or we get a weird pyarrow overflow error
        dataset = dataset.map(
            map_fn,
            cache_file_name=str(cache_file),
            writer_batch_size=100,
            new_fingerprint=new_fingerprint,
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=self.caching_batch_size,
        )
        return dataset

    def _cache_data_for_module(self, module, i):
        self.datasets.append(self._map_and_cache(self.image_file_and_caption_dataset, self.model.get_dataset_map_fn(module, self.size_bucket), cache_file_prefix=f'module_{i}_', new_fingerprint_args=[i]))

    def __getitem__(self, idx):
        if DEBUG:
            print(Path(self.image_file_and_caption_dataset[idx]['image_file']).stem)
        ret = {}
        for ds in self.datasets:
            ret.update(ds[idx])
        return ret

    def __len__(self):
        return len(self.image_file_and_caption_dataset)


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

# Outermost dataset object that the caller uses. Contains multiple ConcatenatedBatchedDataset. Responsible
# for returning the correct batch for the process's data parallel rank. Calls model.prepare_inputs so the
# returned tuple of tensors is whatever the model needs.
class Dataset:
    def __init__(self, dataset_config, model, regenerate_cache=False, caching_batch_size=1):
        super().__init__()
        self.model = model
        self.post_init_called = False
        self.eval_quantile = None
        res = dataset_config['resolution']

        if dataset_config.get('enable_bucket', False):
            size_buckets = make_size_buckets(res, dataset_config.get('min_bucket_reso'), dataset_config.get('max_bucket_reso'), dataset_config.get('bucket_reso_steps'))
        else:
            size_buckets = [(res, res)]

        datasets_by_size_bucket = defaultdict(list)
        for directory_config in dataset_config['directory']:
            size_bucket_to_filepaths = self._split_into_size_buckets(directory_config['path'], size_buckets)
            for size_bucket, filepaths in size_bucket_to_filepaths.items():
                datasets_by_size_bucket[size_bucket].append(
                    SizeBucketDataset(filepaths, directory_config, size_bucket, model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)
                )

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

    def _split_into_size_buckets(self, path, size_buckets):
        size_bucket_to_filepaths = defaultdict(list)
        path = Path(path)
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
                logger.warning(f'Image file {image_file} does not have corresponding caption file. Skipping.')
                continue
            size_bucket = self._find_closest_size_bucket(image_file, size_buckets)
            if size_bucket:
                size_bucket_to_filepaths[size_bucket].append((str(image_file), str(caption_file)))
        return size_bucket_to_filepaths

    def _find_closest_size_bucket(self, image_file, size_buckets):
        try:
            pil_img = Image.open(image_file)
        except Exception:
            logger.warning(f'Image file {image_file} could not be opened. Skipping.')
            return None
        width, height = pil_img.size
        ar = width / height
        best_size_bucket = None
        best_ar_diff = float('inf')
        for size_bucket in size_buckets:
            bucket_ar = size_bucket[0] / size_bucket[1]
            ar_diff = abs(bucket_ar - ar)
            if ar_diff < best_ar_diff:
                best_ar_diff = ar_diff
                best_size_bucket = size_bucket
        return best_size_bucket


# Helper class to make caching multiple datasets more efficient by moving
# models to GPU as few times as needed.
class DatasetManager:
    def __init__(self, model):
        self.model = model
        self.datasets = []

    def register(self, dataset):
        for bucket in dataset.buckets:
            for ds in bucket.datasets:
                self.datasets.append(ds)

    def cache(self):
        with zero_first():
            self._cache()

    @torch.no_grad()
    def _cache(self):
        for i, module in enumerate(self.model.get_modules()):
            if is_main_process():
                module.to('cuda')
            for ds in self.datasets:
                ds._cache_data_for_module(module, i)
            module.to('cpu')
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