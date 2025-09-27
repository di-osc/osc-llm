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
        """High-level convenience wrapper around tokenizer + causal LM runtime."""
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
        """Chat API that renders messages via chat template and generates reply."""
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

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> str:
        """One-shot text generation for a single prompt."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        token_ids = self.tokenizer.encode(prompt).tolist()
        return self.tokenizer.decode(
            self.model.batch([token_ids], [sampling_params])[0]
        )

    def batch_generate(
        self,
        prompts: List[str],
        sampling_params: List[SamplingParams] | None = None,
    ) -> List[str]:
        """Generate completions for a batch of prompts."""
        if sampling_params is None:
            sampling_params = [SamplingParams() for _ in prompts]
        batch_token_ids = [self.tokenizer.encode(prompt).tolist() for prompt in prompts]
        batch_completion_token_ids = self.model.batch(batch_token_ids, sampling_params)
        return [
            self.tokenizer.decode(token_ids) for token_ids in batch_completion_token_ids
        ]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = True,
        add_generate_prompt: bool = True,
    ) -> str:
        """Expose chat templating for external inspection or debugging."""
        return self.tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            add_generate_prompt=add_generate_prompt,
        )
