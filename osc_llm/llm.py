from .models import load_causal_lm


class LLM:
    def __init__(
        self,
        checkpoint_dir: str,
        gpu_memory_utilization: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint_dir: the directory of the checkpoint
            gpu_memory_utilization: the GPU memory utilization
            device: the device to use
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = load_causal_lm(checkpoint_dir)
        self.model.setup(gpu_memory_utilization=gpu_memory_utilization, device=device)

    def generate(
        self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.6
    ):
        from osc_transformers import SamplingParams

        outputs = self.model.batch(
            [prompt],
            [
                SamplingParams(
                    max_generate_tokens=max_new_tokens, temperature=temperature
                )
            ],
        )
        return outputs[0]

    def stream(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.6):
        from osc_transformers import SamplingParams

        outputs = self.model.stream(
            prompt=prompt,
            sampling_params=SamplingParams(
                max_generate_tokens=max_new_tokens, temperature=temperature
            ),
        )
        return outputs
