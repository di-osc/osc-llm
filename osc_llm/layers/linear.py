import torch.nn as nn
import torch
import torch.nn.functional as F
from ..config import registry
from ..utils import find_multiple
import math



@registry.layers.register("Linear")
def Linear(n_in: int, n_out: int, bias: bool = True):
    return nn.Linear(in_features=n_in, out_features=n_out, bias=bias)


@registry.layers.register("WeightOnlyInt8Linear")
class WeightOnlyInt8Linear(nn.Module):
    """用于量化的Linear层，只包含weight和scales两个参数"""
    __contains__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
    

@registry.layers.register("WeightOnlyInt4Linear")
class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
            self, 
            in_features: int, 
            out_features: int,
            bias=False,  
            groupsize: int = 128, 
            inner_k_tiles: int = 8,
    ) -> None:
        super().__init__()
        self.padding = not self._check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return self._linear_forward_int4(input,
                                        self.weight, 
                                        self.scales_and_zeros, 
                                        self.out_features, 
                                        self.groupsize)
        
    def _check_linear_int4_k(self, in_features: int, groupsize: int, inner_k_tiles: int) -> bool:
        """check if the input features are compatible with the linear int4 kernel

        Args:
            in_features (int): the number of input features
            groupsize (int): the group size
            inner_k_tiles (int): the number of inner k tiles
        """
        return in_features % (inner_k_tiles * 16) == 0 and in_features % groupsize == 0
    
    def _linear_forward_int4(self, x, weight_int4pack, scales_and_zeros, out_features, groupsize):
        origin_x_size = x.size()
        x = x.reshape(-1, origin_x_size[-1])
        c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
        new_shape = origin_x_size[:-1] + (out_features,)
        c = c.reshape(new_shape)
        return c
    
    
class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: int = 1,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        assert r >= 0, "r must be greater than or equal to 0"
        self.r = r
        self.alpha = alpha
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, **kwargs)
        if r > 0:
            self.dropout = nn.Dropout(dropout)
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, r))
            self.scale = self.alpha / self.r
            self.reset_parameters()
        self.merged = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.linear(x)
        if self.merged or self.r == 0:
            return x1
        x2 = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
        return x1 + x2
        
    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
    def get_delta_w(self):
        return (self.lora_B @ self.lora_A )* self.scale
    
    def merge(self):
        if self.r == 0:
            return
        self.linear.weight.data += self.get_delta_w()
        self.merged = True