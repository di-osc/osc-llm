import torch.nn as nn
import torch
import torch.nn.functional as F
from ..config import registry


@registry.layers.register("SiLU")
def SiLU(inplace: bool = False) -> nn.Module:
    return nn.SiLU(inplace=inplace)


@registry.layers.register("GeLU")
def GELU() -> nn.Module:
    return nn.GELU()


@registry.layers.register("ReLU")
def ReLU(inplace: bool = False) -> nn.Module:
    return nn.ReLU(inplace=inplace)


@registry.layers.register("Sigmoid")
def Sigmoid() -> nn.Module:
    return nn.Sigmoid()


@registry.layers.register("Tanh")
def Tanh() -> nn.Module:
    return nn.Tanh()


@registry.layers.register("Softmax")
def Softmax(dim: int) -> nn.Module:
    return nn.Softmax(dim=dim)


@registry.layers.register("SiluAndMul")
class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.silu(x) * y
