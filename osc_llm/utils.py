from typing import Optional
import statistics
import uuid

import torch
from wasabi import msg

from .config import registry, Config


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_chat_template(name) -> Optional[Config]:
    """在一个checkpoint的名称中获取chat模板"""
    for k, v in registry.chat_templates.get_all().items():
        if k in name:
            return v
    return None


@torch.no_grad()
def benchmark(model, num_iters=10, **inputs):
    """Runs the model on the input several times and returns the median execution time."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        start.record()
        model(**inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    msg.info(f"Number of iterations: {num_iters}.")
    all_time = sum(times)
    msg.info(f"Total time: {all_time:.4f} s")
    mean_time = statistics.mean(times)
    msg.info(f"Mean time: {mean_time:.4f} s")
    median_time = statistics.median(times)
    msg.info(f"Median time: {median_time:.4f} s")


def get_default_supported_precision(training: bool = False) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning_fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (
        torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    ):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def get_model_size(model: torch.nn.Module, contains_embedding: bool = False) -> int:
    """Get the size of a model in bytes.

    Args:
        model (torch.nn.Module): the model to get the size of.
        contains_embedding (bool, optional): whether the model contains an embedding layer. Defaults to False.

    Returns:
        int: the size of the model in bytes.
    """
    size = 0
    for param in model.parameters():
        size += param.numel() * param.element_size()
    if contains_embedding:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                size += (
                    module.num_embeddings
                    * module.embedding_dim
                    * module.weight.element_size()
                )
    return size
