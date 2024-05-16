import torch
from abc import ABC, abstractmethod


class Sampler(ABC):
    """Sampler can be used to sample from a distribution given logits to get ids."""

    @abstractmethod
    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def probs_to_ids(self, probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_config(self, section: str = "sampler"):
        raise NotImplementedError

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self.logits_to_probs(logits)
        return self.probs_to_ids(probs)

    def multinomial_sample_one(self, probs: torch.Tensor) -> torch.Tensor:
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True).to(dtype=torch.int)
