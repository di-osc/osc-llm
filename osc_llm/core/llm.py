from __future__ import annotations
import atexit
from typing import Generator
from time import perf_counter

from tqdm.auto import tqdm
import torch

from .sequence import Sequence
from .scheduler import Scheduler
from .model_runner import ModelRunner
from .sampling_params import SamplingParams
from ..tokenizer import Tokenizer
from .config import LLMConfig


class LLM:
    def __init__(
        self,
        model: str,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = False,
        kvcache_block_size: int = 256,
        num_kvcache_blocks: int = -1,
    ):
        self.config = LLMConfig(
            model=model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            kvcache_block_size=kvcache_block_size,
            num_kvcache_blocks=num_kvcache_blocks,
        )
        self.model_runner = ModelRunner(self.config)
        self.tokenizer = Tokenizer(checkpoint_dir=self.config.model)
        self.config.eos = self.tokenizer.eos_id
        self.scheduler = Scheduler(config=self.config)
        atexit.register(self.exit)

    def exit(self):
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt).tolist()
        else:
            token_ids = prompt
        seq = Sequence(token_ids=token_ids, sampling_params=sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> list[str]:
        if sampling_params is None:
            sampling_params = [SamplingParams() for _ in prompts]
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {
                "text": self.tokenizer.decode(torch.tensor(token_ids)),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs

    def stream(
        self, prompt: str, sampling_params: SamplingParams = None
    ) -> Generator[str, None, None]:
        assert isinstance(prompt, str)
        if sampling_params is None:
            sampling_params = SamplingParams()

        def token_generator():
            token_ids = self.tokenizer.encode(prompt).tolist()
            seq = Sequence(token_ids=token_ids, sampling_params=sampling_params)
            self.scheduler.add(seq)

            # 跟踪序列的完成token数量，用于检测新生成的token
            last_completion_tokens = 0

            while not seq.is_finished:
                # 执行一步推理
                seqs, is_prefill = self.scheduler.schedule()
                token_ids = self.model_runner.call("run", seqs, is_prefill)
                self.scheduler.postprocess(seqs, token_ids)

                # 检查我们的序列是否有新的token生成
                if (
                    seq in seqs
                    and len(seq.completion_token_ids) > last_completion_tokens
                ):
                    # 获取新生成的token
                    new_tokens = seq.completion_token_ids[last_completion_tokens:]
                    last_completion_tokens = len(seq.completion_token_ids)

                    # 将新token转换为torch.Tensor并yield
                    for token_id in new_tokens:
                        yield torch.tensor([token_id], dtype=torch.int)

        # 使用tokenizer的decode_stream方法来处理token流
        return self.tokenizer.decode_stream(token_generator())
