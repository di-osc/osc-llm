from typing import Optional
import statistics
import uuid

import torch
from wasabi import msg
from confection import Config

from .registry import Registry


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_chat_template(name) -> Optional[Config]:
    """在一个checkpoint的名称中获取chat模板"""
    for k, v in Registry.chat_templates.get_all().items():
        if k in name:
            return v
    return None


def random_uuid() -> str:
    return str(uuid.uuid4())


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
