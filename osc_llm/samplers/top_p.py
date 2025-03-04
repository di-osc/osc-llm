from ..config import registry, Config
from .base import Sampler
import torch


@registry.samplers.register("TopP")
class TopP(Sampler):
    def __init__(
        self,
        p: float = 0.5,
    ):
        super().__init__()
        self.p = p

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        probs, self.indices = logits.softmax(dim=-1).sort(descending=True)
        nucleus_mask = probs.cumsum(dim=-1) < self.p
        nucleus_mask = torch.cat(
            [
                nucleus_mask.new_ones(nucleus_mask.shape[:-1] + (1,)),
                nucleus_mask[..., :-1],
            ],
            dim=-1,
        )
        probs = probs * nucleus_mask
        probs /= probs.sum(dim=-1, keepdim=True)
        return probs

    def probs_to_ids(self, probs: torch.Tensor) -> torch.Tensor:
        descending_idx = self.multinomial_sample_one(probs)
        # 得到真实的下标
        return self.indices.gather(-1, descending_idx)

    def get_config(self, section: str = "sampler"):
        config_str = f"""
        [{section}]
        @samplers = TopP
        p = {self.p}"""
        config = Config().from_str(config_str)
        return config
