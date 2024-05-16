from ..config import registry, Config
from .base import Sampler
import torch


@registry.samplers.register("TopK")
class TopK(Sampler):
    def __init__(
        self,
        k: int = 10,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = 0.001 if temperature <= 0 else temperature
        self.k = k

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature
        values, indices = torch.topk(logits, min(self.k, logits.shape[-1]), dim=-1)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, indices, values)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def probs_to_ids(self, probs: torch.Tensor) -> torch.Tensor:
        ids = self.multinomial_sample_one(probs=probs)
        return ids

    def get_config(self, section: str = "sampler"):
        config_str = f"""
        [{section}]
        @samplers = TopK
        k = {self.k}
        temperature = {self.temperature}"""
        config = Config().from_str(config_str)
        return config
