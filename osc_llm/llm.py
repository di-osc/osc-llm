from typing import List, Dict, Generator

from osc_transformers import SamplingParams

from .models import load_causal_lm
from .tokenizer import Tokenizer


class LLM:
    def __init__(
        self,
        checkpoint_dir: str,
        gpu_memory_utilization: float = 0.5,
        device: str = "cuda",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
        self.model = load_causal_lm(checkpoint_dir)
        self.model.setup(
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            eos_id=self.tokenizer.eos_id,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: SamplingParams | None = None,
        enable_thinking: bool = True,
        stream: bool = False,
    ) -> Dict[str, str] | Generator[str, None, None]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        prompt = self.tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking
        )
        token_ids = self.tokenizer.encode(prompt).tolist()
        if stream:
            return self.tokenizer.decode_stream(
                self.model.stream(token_ids, sampling_params)
            )
        else:
            content = self.tokenizer.decode(
                self.model.batch([token_ids], [sampling_params])[0]
            )
            thinking_content, content = self.tokenizer.split_thinking_content(content)
            return {
                "role": "assistant",
                "content": content,
                "thinking_content": thinking_content,
            }

    def batch(
        self,
        messages: List[List[Dict[str, str]]],
        sampling_params: List[SamplingParams] | None = None,
        enable_thinking: bool = True,
    ) -> List[Dict[str, str]]:
        if sampling_params is None:
            sampling_params = [SamplingParams() for _ in messages]
        batch_prompts = [
            self.tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking
            )
            for messages in messages
        ]
        batch_token_ids = [
            self.tokenizer.encode(prompt).tolist() for prompt in batch_prompts
        ]
        batch_completion_token_ids = self.model.batch(batch_token_ids, sampling_params)
        batch_contents = [
            self.tokenizer.decode(token_ids) for token_ids in batch_completion_token_ids
        ]
        batch_results = []
        for content in batch_contents:
            thinking_content, content = self.tokenizer.split_thinking_content(content)
            batch_results.append(
                {
                    "role": "assistant",
                    "content": content,
                    "thinking_content": thinking_content,
                }
            )
        return batch_results
