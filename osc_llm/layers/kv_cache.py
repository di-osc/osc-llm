import torch.nn as nn
import torch
from ..config import registry
from typing import Tuple, Optional


class KVCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def setup(
        self,
        k_shape: Tuple[int],
        v_shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """构建KVCache的缓存

        Args:
            k_shape (Tuple[int]): k缓存的形状, 通常为(batch_size, n_heads, seq_length, head_size)
            v_shape (Tuple[int]): v缓存的形状, 通常为(batch_size, n_heads, seq_length, head_size)
            dtype (Optional[torch.dtype], optional): 数据类型.
            device (Optional[torch.device], optional): 缓存存储设备.
        """
        self.register_buffer(
            "k_cache",
            torch.zeros(k_shape, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(v_shape, dtype=dtype, device=device),
            persistent=False,
        )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        input_pos: torch.Tensor,
        copy_dim: torch.Tensor,
    ):
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

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        input_pos: torch.Tensor,
        copy_dim: int = -2,
    ):
        """更新KVCache的缓存

        Args:
            k (torch.Tensor): 当前的k
            v (torch.Tensor): 当前的v
            input_pos (torch.Tensor): 输入的位置
            copy_dim (int, optional): 复制的维度. Defaults to 2.

        Returns:
            Tuple[Tensor, Tensor]: 更新后的k和v.
        """
        assert hasattr(self, "k_cache") and hasattr(
            self, "v_cache"
        ), "KVCache must be setup before updating. Use `setup` method to setup KVCache"

        self.k_cache: torch.Tensor = self.k_cache.to(k.dtype)
        self.v_cache: torch.Tensor = self.v_cache.to(v.dtype)

        bs = k.size(0)
        k = batched_index_copy_(self.k_cache[:bs, ...], copy_dim, input_pos, k)
        v = batched_index_copy_(self.v_cache[:bs, ...], copy_dim, input_pos, v)
        return k, v


def batched_index_copy_(t: torch.Tensor, dim, idx, val):
    """Index copy for batched tensor, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim {dim=}"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t
