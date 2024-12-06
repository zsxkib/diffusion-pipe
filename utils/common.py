from contextlib import contextmanager
import gc
import time

import torch
import deepspeed.comm.comm as dist
import imageio


DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
VIDEO_EXTENSIONS = set(x.extension for x in imageio.config.video_extensions)
AUTOCAST_DTYPE = None


def get_rank():
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@contextmanager
def zero_first():
    if not is_main_process():
        dist.barrier()
    yield
    if is_main_process():
        dist.barrier()


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def log_duration(name):
    start = time.time()
    try:
        yield
    finally:
        print(f'{name}: {time.time()-start:.3f}')