import torch.nn as nn 
import torch
from ..config import registry
from typing import Tuple, Optional



class KVCache(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
    
    def setup(self, k_shape: Tuple[int], v_shape: Tuple[int], dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        """构建KVCache的缓存

        Args:
            k_shape (Tuple[int]): k缓存的形状, 通常为(batch_size, n_heads, seq_length, head_size)
            v_shape (Tuple[int]): v缓存的形状, 通常为(batch_size, n_heads, seq_length, head_size)
            dtype (Optional[torch.dtype], optional): 数据类型.
            device (Optional[torch.device], optional): 缓存存储设备.
        """
        self.register_buffer("k_cache", torch.zeros(k_shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v_cache", torch.zeros(v_shape, dtype=dtype, device=device), persistent=False)
    
    def update(self, k: torch.Tensor, v: torch.Tensor, input_pos: torch.Tensor, copy_dim: torch.Tensor):
        raise NotImplementedError
    
    def reset_parameter(self):
        torch.nn.init.zeros_(self.k_cache)
        torch.nn.init.zeros_(self.v_cache)
        
    def reset_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        
    def get_max_length(self):
        raise NotImplementedError
    
    def get_seq_length(self):
        raise NotImplementedError
    
    

@registry.layers.register("StaticKVCache")
class StaticKVCache(KVCache):
    """提前分配好形状的KVCache,在模型推理过程中形状不会变化,如果超出最大长度,则会报错。
    缓存的形状一般为: (batch_size, n_heads, seq_length, head_size)
    """
    def __init__(self) -> None:
        super().__init__()
        
    def update(self, k: torch.Tensor, v: torch.Tensor, input_pos: torch.Tensor, copy_dim: int = 2):
        """更新KVCache的缓存

        Args:
            k (torch.Tensor): 当前的k
            v (torch.Tensor): 当前的v
            input_pos (torch.Tensor): 输入的位置
            copy_dim (int, optional): 复制的维度. Defaults to 2.

        Returns:
            Tuple[Tensor, Tensor]: 更新后的k和v.
        """
        assert hasattr(self, "k_cache") and hasattr(self, "v_cache"), "KVCache must be setup before updating. Use `setup` method to setup KVCache"
            
        
        self.k_cache: torch.Tensor = self.k_cache.to(k.dtype)
        self.v_cache: torch.Tensor = self.v_cache.to(v.dtype)
        
        k = self.k_cache.index_copy_(dim=copy_dim, index=input_pos, source=k)
        v = self.v_cache.index_copy_(dim=copy_dim, index=input_pos, source=v)
        return k, v