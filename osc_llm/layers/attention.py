import torch.nn as nn
import torch
from typing import Optional
from ..config import registry
from .kv_cache import KVCache
import math
    
    
@registry.layers.register("CausalSelfAttention")  
class CausalSelfAttention(nn.Module):
    """融合了RoPE,MQA,GQA,MHA的因果自注意力机制层"""
    
    # 当`n_heads=4`时MHA,GQA,MQA的区别:
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    
    def __init__(
        self, 
        n_in: int, 
        n_heads: int,
        q_bias: bool = False,
        k_bias: bool = False,
        v_bias: bool = False,
        o_bias: bool = False,
        n_query_groups: Optional[int] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        super().__init__()
        
        assert n_in % n_heads == 0, f"dim {n_in} must be divisible by n_heads {n_heads}"
        
        self.n_heads = n_heads
        self.head_size = n_in // n_heads
        self.n_query_groups = n_query_groups if n_query_groups else n_heads
        
        self.q_proj = nn.Linear(n_in, self.n_heads * self.head_size, bias=q_bias)
        self.k_proj = nn.Linear(n_in, self.n_query_groups * self.head_size, bias=k_bias)
        self.v_proj = nn.Linear(n_in, self.n_query_groups * self.head_size, bias=v_bias)
        self.o_proj = nn.Linear(n_in, n_in, bias=o_bias)
        
        self.kv_cache: KVCache = kv_cache


    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None
    ):
        
        B, L, D = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        q: torch.Tensor = self.q_proj(x).reshape(B, L, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k: torch.Tensor = self.k_proj(x).reshape(B, L, self.n_query_groups, self.head_size).permute(0, 2, 1, 3)
        v: torch.Tensor = self.v_proj(x).reshape(B, L, self.n_query_groups, self.head_size).permute(0, 2, 1, 3)
        
        if (cos is not None) and (sin is not None):
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            
        if input_pos is not None:
            if not self.kv_cache:
                raise TypeError("current attention layer does not support kv_cache, please set `kv_cache` in the layer init")
            if not hasattr(self.kv_cache, "k_cache") or not hasattr(self.kv_cache, "v_cache"):
                raise TypeError("You need to call `model.setup_kv_cache()`")
            k, v = self.kv_cache.update(input_pos=input_pos, k=k, v=v, copy_dim=2)
            
        if self.n_query_groups != 1 and self.n_query_groups != self.n_heads:  # doing this would require a full kv cache with MQA (inefficient!)
            # for MHA this is a no-op
            k = k[:,:,None,:,:].expand(-1, -1, self.n_heads // self.n_query_groups, -1, -1).reshape(B, self.n_heads, -1, self.head_size)
            v = v[:,:,None,:,:].expand(-1, -1, self.n_heads // self.n_query_groups, -1, -1).reshape(B, self.n_heads, -1, self.head_size)

        o = self.scaled_dot_product_attention(q, k, v, mask=attention_mask)

        o = o.reshape(B, L, D)

        o = self.o_proj(o)

        return o

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)
    
    def setup_kv_cache(self, 
                     batch_size: int, 
                     max_seq_length: int, 
                     device: Optional[torch.device] = None, 
                     dtype: Optional[torch.dtype] = None) -> None:
        n_heads = self.n_query_groups
        k_shape = (batch_size, n_heads, max_seq_length, self.head_size)
        v_shape = (batch_size, n_heads, max_seq_length, self.head_size)
        self.kv_cache.setup(k_shape, v_shape, dtype=dtype, device=device)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(x.dtype) 