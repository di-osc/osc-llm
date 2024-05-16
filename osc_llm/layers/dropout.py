from ..config import registry
import torch.nn as nn


@registry.layers.register("Dropout")
def Dropout(p: float = 0.1, inplace: bool = False):
    return nn.Dropout(p=p, inplace=inplace)
