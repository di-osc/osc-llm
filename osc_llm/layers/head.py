import torch.nn as nn
from ..config import registry



@registry.layers.register("LMHead")
class LMHead(nn.Module):
    def __init__(self, n_in: int, n_out: int, bias: bool) -> None:
        super().__init__()
        self.predictor = nn.Linear(n_in, n_out, bias=bias)
        
    def forward(self, x):
        return self.predictor(x)