import os
import time
from random import randint
from pathlib import Path
from typing import Literal, Optional

from jsonargparse import CLI
from wasabi import msg


def download_model(
    repo_id: str,
    save_dir: str = "./checkpoints",
    force_download: bool = False,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    endpoint: Literal["hf", "hf-mirror", "modelscope"] = "hf-mirror",
):
    """
    Download a model from the Hugging Face Hub or ModelScope.

    Args:
        repo_id: The ID of the model to download.
        save_dir: The directory to save the downloaded model.
        force_download: Whether to force the download of the model.
        access_token: The access token to use for the Hugging Face Hub.
        endpoint: The endpoint to use for the Hugging Face Hub.
    """
    directory = Path(save_dir, repo_id)

    if endpoint == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=directory,
        )
    else:
        if endpoint == "hf-mirror":
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id,
            local_dir=directory,
            force_download=force_download,
            token=access_token,
        )


def bench(
    model: str,
    num_seqs: int = 64,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
    temperature: float = 0.6,
    max_num_batched_tokens: int = 16384,
    max_num_seqs: int = 512,
    max_model_len: int = 4096,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = 0.5,
    kvcache_block_size: int = 256,
    num_kvcache_blocks: int = -1,
):
    """
    Benchmark the osc model.

    Args:
        model_path: Path to the osc model directory containing the checkpoint.
        num_seqs: The number of sequences to benchmark.
        max_input_len: The maximum input length.
        max_output_len: The maximum output length.
        temperature: The temperature of the sampling.
        max_num_batched_tokens: The maximum number of batched tokens.
        max_num_seqs: The maximum number of sequences.
        max_model_len: The maximum model length.
        enforce_eager: Whether to enforce eager mode.
        gpu_memory_utilization: The GPU memory utilization.
        kvcache_block_size: The KV cache block size.
        num_kvcache_blocks: The number of KV cache blocks.
    """
    from random import seed

    seed(0)
    from .core import LLM, SamplingParams

    llm = LLM(
        model=model,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        kvcache_block_size=kvcache_block_size,
        num_kvcache_blocks=num_kvcache_blocks,
    )
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=temperature,
            ignore_eos=True,
            max_tokens=randint(100, max_output_len),
        )
        for _ in range(num_seqs)
    ]
    with msg.loading("Warming up model..."):
        _ = llm.generate(["Benchmark: "], sampling_params, use_tqdm=False)
    with msg.loading("Benchmark Throughput..."):
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = time.time() - t
        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t
    with msg.loading("Benchmark First Token Latency..."):
        latencys = []
        for i in range(10):
            latency = 0
            start = time.time()
            for _ in llm.stream(prompt="Benchmark: "):
                if latency == 0:
                    latency = time.time() - start
                    break
            latencys.append(latency)
        latency = sum(latencys) / len(latencys)
        latency = latency * 1000
    header = ("model", "Throughput", "First token latency")
    data = [(f"{llm.model_runner.hf_architecture}", f"{throughput:.2f}tok/s", f"{latency:.2f}ms")]
    msg.table(data=data, header=header, divider=True, aligns=("c", "c", "c"), title="Benchmark Result")


def serve_openai(
    model_path: str,
    port: int = 8000,
):
    """
    Serve the osc model as an OpenAI API.

    Args:
        model_path: Path to the osc model directory containing the checkpoint.
        port: The port to serve the OpenAI API.
    """
    from .servers.openai import main as openai_main

    openai_main(model_path, port)


commands = {
    "download": download_model,
    "serve": serve_openai,
    "bench": bench,
}


def run():
    CLI(components=commands)


if __name__ == "__main__":
    run()
