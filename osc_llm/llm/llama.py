from dataclasses import dataclass
import torch.nn as nn
from typing import Tuple, Optional, Any, List
import torch
import math
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from ..layer import RMSNorm


RoPECache = Tuple[torch.Tensor, torch.Tensor]


FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


@dataclass
class LlamaConfig:
    org: str = "OSC-AI"
    name: str = "llama-7B"
    vocab_size: int = 55296
    block_size: int = 4096
    padding_multiple: int = 64
    n_layer: int = 32
    n_embd: int = 4096
    n_head: int = 32
    bias: bool = False
    rotary_percentage: float = 1.0
    n_query_groups: Optional[int] = None
    norm_eps: float = 1e-5
    intermediate_size: Optional[int] = 11008
    condense_ratio: int = 1.0

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_head == 0
        
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

name_to_config = {"7B": {"name": "llama-7B",
                         "block_size": 4096, 
                         "n_layer": 32, 
                         "n_embd": 4096, 
                         "n_head": 32, 
                         "n_query_groups": 32, 
                         "intermediate_size": 11008},
                  "1B": {"name": "llama-1B",
                         "block_size": 2048,
                         "n_layer": 32,
                         "n_embd": 4096,
                         }}


class LlamaMLP(torch.nn.Module):
    def __init__(self, n_embd: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.fc_1 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.fc_2 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.proj = nn.Linear(intermediate_size, n_embd, bias=bias)
        
    def forward(self, x):
        x1 = self.fc_1(x)
        x2 = self.fc_2(x)
        x = nn.functional.silu(x1) * x2
        return self.proj(x)
    
class KVCache(nn.Module):
    def __init__(self, 
                 k_shape: Tuple[int, int, int, int],
                 v_shape: Tuple[int, int, int, int],
                 device: torch.device,
                 dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)
        
    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(config.vocab_size, config.n_embd), 
                                              h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                                              ln_f=RMSNorm(config.n_embd, eps=config.norm_eps)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.mask_cache: Optional[torch.Tensor] = None
        
        # 默认最大序列长度为block_size,并且RoPE缓存的长度也为block_size
        self.max_seq_length = config.block_size
        
    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `model.apply(model._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        B, T = idx.size()
        if self.max_seq_length < T:
            raise ValueError(f"Cannot attend to {T} tokens when max_seq_length is only {self.max_seq_length}")
        
        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `model.build_kv_caches()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)
        

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(LlamaConfig.from_name(name, **kwargs))
    
    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length
    
    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """设置模型输入序列的最大长度,并且重置RoPE缓存.
        
        注意:
        - 因为kv缓存无法在设置max_seq_length的时候确定是否使用(训练或者推理),所以不在这里设置
        """
        assert value <= self.config.block_size, f"Cannot attend to {value}, block size is only {self.config.block_size}"
        self._max_seq_length = value
        ## 当设置最大长度的时候, 重置RoPE缓存,以节省内存消耗
        if not hasattr(self, "cos") or not hasattr(self, "sin"):
            cos, sin = self.build_rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        ## 如果当前的RoPE缓存长度不等于设置的最大长度,则重置RoPE缓存
        elif self.cos.size(0) != value or self.sin.size(0) != value:
            cos, sin = self.build_rope_cache(self.cos.device)
            self.cos = cos
            self.sin = sin 
                

    def build_rope_cache(self, device: Optional[torch.device] = None) -> RoPECache:
        return build_rope_cache(seq_len=self.max_seq_length,
                                n_elem=self.config.head_size,
                                dtype=torch.get_default_dtype(),
                                device=device,
                                condense_ratio=self.config.condense_ratio)

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)
    

    def build_kv_caches(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.device] = None) -> List[KVCache]:
        
        for block in self.transformer.h:
            block.attn.build_kv_cache(batch_size, self.max_seq_length, device, dtype)
        
        if self.mask_cache is None or self.mask_cache.size(3) != self.max_seq_length:
            ones = torch.ones((self.config.block_size, self.max_seq_length), device=device, dtype=torch.bool)
            self.mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)
            
            
    def clear_kv_caches(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None
    
    
class Block(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(size=config.n_embd, eps=config.norm_eps)
        self.attn = LlamaAttention(config)
        self.norm_2 = RMSNorm(size=config.n_embd, eps=config.norm_eps)
        self.mlp = LlamaMLP(n_embd=config.n_embd, intermediate_size=config.intermediate_size, bias=config.bias)

    def forward(self,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        h = self.attn(x=self.norm_1(x), cos=cos, sin=sin, mask=mask, input_pos=input_pos)
        # 注意力层残差
        x = x + h
        # MLP残差
        x = x + self.mlp(self.norm_2(x))
        return x 
        
       
class LlamaAttention(nn.Module):
    # Example with `n_head=4`
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
    
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_size, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_query_groups * config.head_size, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_query_groups * config.head_size, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_query_groups * config.head_size, bias=config.bias)
        
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(self,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[KVCache]]:
        
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.q_proj(x).reshape(B, T, self.config.n_head, self.config.head_size).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.config.n_query_groups, self.config.head_size).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.config.n_query_groups, self.config.head_size).permute(0, 2, 1, 3)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # repeat k and v if necessary
        if self.config.n_query_groups != 1 and self.config.n_query_groups != self.config.n_head:  # doing this would require a full kv cache with MQA (inefficient!)
            # for MHA this is a no-op
            k = k.expand(B, self.config.n_query_groups, T, self.config.n_head // self.config.n_query_groups, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, T, self.config.n_head // self.config.n_query_groups, self.config.head_size)
        
        
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `model.build_kv_caches()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.o_proj(y)

        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            # flash-attn requires (B, T, nh, hs)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)
    
    def build_kv_cache(self, 
                     batch_size: int, 
                     max_seq_length: int, 
                     device: Optional[torch.device] = None, 
                     dtype: Optional[torch.dtype] = None) -> None:
        n_head = 1 if self.config.n_query_groups == 1 else self.config.n_head
        k_shape = (batch_size, n_head, max_seq_length, self.config.head_size)
        v_shape = (batch_size, n_head, max_seq_length, self.config.head_size)
        self.kv_cache = KVCache(k_shape, v_shape, device, dtype)
        
        
        
def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
        
        