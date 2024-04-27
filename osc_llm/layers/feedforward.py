from ..config import registry
import torch.nn as nn
from copy import deepcopy
import torch


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
    
    
@registry.layers.register("MoE")
class MoE(nn.Module):
    def __init__(
        self, 
        n_experts: int, 
        n_activated_experts: int,
        n_in: int,
        expert: nn.Module,
        gate_bias: bool = False,
        ) -> None:
        super().__init__()
        self.gate = nn.Linear(n_in, n_experts, bias=gate_bias)
        self.experts = nn.ModuleList(deepcopy(expert) for _ in range(n_experts))
        self.n_activated_experts = n_activated_experts
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.n_activated_experts)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.n_experts, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C), indices