"""
GPU utilities for the merging pipeline.
"""
import torch
import gc


def release_cuda_memory():
    """
    Aggressively releases CUDA memory by running garbage collection
    and clearing the CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
