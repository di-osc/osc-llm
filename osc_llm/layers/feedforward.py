from ..config import registry
import torch.nn as nn
from copy import deepcopy
import torch
from typing import Optional
import torch.nn.functional as F



@registry.layers.register("GLU")
class GLU(nn.Module):
    """门控线性单元
    """
    def __init__(self, 
                 n_in: int, 
                 n_hidden: int,
                 activation: nn.Module,
                 up_bias: bool = False,
                 gate_bias: bool = False,
                 down_bias: bool = False) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_in, n_hidden, bias=up_bias)
        self.gate_proj = nn.Linear(n_in, n_hidden, bias=gate_bias)
        self.down_proj = nn.Linear(n_hidden, n_in, bias=down_bias)
        self.activation = activation
        
    def forward(self, x):
        x1 = self.up_proj(x)
        x2 = self.gate_proj(x)
        x = x1 * self.activation(x2)
        x = self.down_proj(x)
        return x
    

@registry.layers.register("SwiGLU.v1")
@registry.layers.register("SwiGLU")
def SwiGLU(n_in: int, 
           n_hidden: int,
           up_bias: bool = False,
           gate_bias: bool = False,
           down_bias: bool = False) -> nn.Module:
    """Swish激活函数的门控线性单元
    """
    return GLU(n_in=n_in,
               n_hidden=n_hidden,
               activation=nn.SiLU(),
               up_bias=up_bias,
               gate_bias=gate_bias,
               down_bias=down_bias)
    

@registry.layers.register("SwiGLU.v2")
class SwiGLUV2(nn.Module):
    """Swish激活函数的门控线性单元另外一种实现方式,
    
    与v1版本的区别: 
    - 原本的up_proj和gate_proj合并为up_gate_proj
    - 前向传播时, up_gate_proj的输出先前后切分为up和gate两部分
    """
    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        up_gate_bias: bool = False,
        down_bias: bool = False
    ):
        super().__init__()
        
        self.up_gate_proj = nn.Linear(n_in, n_hidden * 2, bias=up_gate_bias)
        self.down_proj = nn.Linear(n_hidden, n_in, bias=down_bias)

    def forward(self, x):
        x_gate, x_up = torch.chunk(self.up_gate_proj(x), 2, dim=-1)
        x = F.silu(x_gate) * x_up
        x = self.down_proj(x)
        return x
    
    
@registry.layers.register("GeGLU")
def GeGLU(n_in: int, 
          n_hidden: int,
          up_bias: bool = False,
          gate_bias: bool = False,
          down_bias: bool = False) -> nn.Module:
    """GELU激活函数的门控线性单元
    """
    return GLU(n_in=n_in,
               n_hidden=n_hidden,
               activation=nn.GELU(),
               up_bias=up_bias,
               gate_bias=gate_bias,
               down_bias=down_bias)
        
    
@registry.layers.register("SparseMoe")
class SparseMoe(nn.Module):
    def __init__(
        self,
        n_experts: int, 
        n_activated_experts: int,
        expert: nn.Module,
        gate: nn.Module,
        norm_probs: bool = True,
        shared_expert: Optional[nn.Module] = None,
        shared_gate: Optional[nn.Module] = None
        ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(deepcopy(expert) for _ in range(n_experts))
        self.n_activated_experts = n_activated_experts
        self.n_experts = n_experts
        self.norm_probs = norm_probs
        self.shared_expert = shared_expert
        self.shared_gate = shared_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        
        logits = self.gate(x)  # (B*T, n_expert)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        probs, indices = torch.topk(probs, self.n_activated_experts, dim=-1)  # probs: (B*T, n_expert_per_token), indices: (B*T, n_expert_per_token)
        if self.norm_probs:
            probs /= probs.sum(dim=-1, keepdim=True)
        probs = probs.to(dtype=x.dtype)
        
        masks = indices.unsqueeze(-1) == torch.arange(self.n_experts, device=x.device) # (B*T, n_expert_per_token, n_expert)
        masks = masks.permute(2, 0, 1) # (n_expert, B*T, n_expert_per_token)
        
        y = torch.zeros_like(x)  # (B*T, C)
        for expert, mask in zip(self.experts, masks):
            # 找出该专家对应的token
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        
        if self.shared_expert is not None:
            y += self.shared_expert(x) * self.shared_gate(x).sigmoid()
            
        return y.reshape(B, T, C)