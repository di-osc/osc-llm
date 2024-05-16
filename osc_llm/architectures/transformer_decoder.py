from ..config import registry
from ..layers.kv_cache import KVCache, StaticKVCache
import torch.nn as nn
from typing import Mapping, Optional, Tuple, List, Any
import torch
from copy import deepcopy


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        attention_norm: nn.Module,
        feedforward: nn.Module,
        feedforward_norm: nn.Module,
        prenorm: bool = True,
    ):
        super().__init__()
        self.attention = attention
        self.attention_norm = attention_norm
        self.feedforward = feedforward
        self.feedforward_norm = feedforward_norm
        self.prenorm = prenorm

    def forward(
        self,
        x,
        input_pos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        if self.prenorm:
            x = (
                self.attention(
                    self.attention_norm(x),
                    input_pos=input_pos,
                    attention_mask=attention_mask,
                    cos=cos,
                    sin=sin,
                )
                + x
            )
            x = x + self.feedforward(self.feedforward_norm(x))
        else:
            x = self.attention_norm(
                self.attention(
                    x,
                    input_pos=input_pos,
                    attention_mask=attention_mask,
                    sin=sin,
                    cos=cos,
                )
                + x
            )
            x = self.feedforward_norm(self.feedforward(x) + x)
        return x


@registry.architectures.register("TransformerDecoder")
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        block_size: int,
        embedding: nn.Module,
        attention: nn.Module,
        feedforward: nn.Module,
        head: nn.Module,
        norm: nn.Module,
        prenorm: bool,
        rope_base: int = 10000,
        kv_cache: Optional[KVCache] = None,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.n_blocks = n_blocks
        self.embedding = embedding
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    attention=deepcopy(attention),
                    attention_norm=deepcopy(norm),
                    feedforward=deepcopy(feedforward),
                    feedforward_norm=deepcopy(norm),
                    prenorm=prenorm,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head_norm = norm if self.prenorm else None
        self.head = head
        self.rope_base = rope_base

        self.block_size = block_size
        self.max_length = block_size

        self.mask_cache: Optional[torch.Tensor] = None

        if kv_cache:
            self.kv_caches = [deepcopy(kv_cache) for _ in range(n_blocks)]

    @property
    def kv_caches(self) -> List[KVCache]:
        return [block.attention.kv_cache for block in self.blocks]

    @kv_caches.setter
    def kv_caches(self, value: List[KVCache]):
        assert len(value) == len(self.blocks), "Number of kv_caches must match number of blocks"
        for block, kv_cache in zip(self.blocks, value):
            block.attention.kv_cache = kv_cache

    @property
    def max_length(self) -> int:
        return self._max_length

    @max_length.setter
    def max_length(self, value: int):
        self._max_length = value
        if self.rope_base:
            if not hasattr(self, "cos") or not hasattr(self, "sin"):
                self.setup_rope_cache(max_length=value)
            elif self.cos.size(0) != value:
                self.setup_rope_cache(max_length=value, device=self.cos.device)

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.max_seq_length = self.block_size

    def setup_kv_cache(
        self,
        batch_size: int,
        max_length: Optional[int] = None,
        kv_cache: Optional[KVCache] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if kv_cache:
            assert isinstance(kv_cache, KVCache), "kv_cache must be an instance of KVCache"
        else:
            kv_cache = StaticKVCache()
        self.kv_caches = [deepcopy(kv_cache) for _ in range(self.n_blocks)]
        if not max_length:
            max_length = self.block_size
        else:
            assert max_length <= self.block_size, "max_length must be less than or equal to block_size"

        for block in self.blocks:
            block.attention.setup_kv_cache(
                batch_size=batch_size,
                max_seq_length=max_length,
                device=device,
                dtype=dtype,
            )

        self.mask_cache = (
            torch.tril(torch.ones((max_length, max_length), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        )

    def setup_rope_cache(self, max_length: int, device: Optional[torch.device] = None) -> None:
        head_size = self.blocks[0].attention.head_size
        cos, sin = build_rope_cache(
            seq_len=max_length,
            n_elem=head_size,
            dtype=torch.get_default_dtype(),
            base=self.rope_base,
            device=device,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None):
        """Forward pass of the TransformerDecoder.

        Args:
            input_ids (torch.Tensor): Input token ids. shape = (batch_size, seq_length)
            input_pos (Optional[torch.Tensor], optional): Input position ids. prefill stage shape = (batch_size, seq_length) decode stage shape = (batch_size, 1). Defaults to None.
        """

        B, L = input_ids.size()

        if self.max_length < L:
            raise ValueError(f"Cannot forward sequence of length {L}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:
            # use rope cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)

            if self.mask_cache is None:
                raise TypeError("You need to call `model.setup_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:L]
            sin = self.sin[:L]
            mask = None

        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x, input_pos=input_pos, cos=cos, sin=sin, attention_mask=mask)

        if self.prenorm:
            x = self.head_norm(x)

        x = self.head(x)

        return x

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True):
        # 保证在用torch.device('meta')构建模型后, 可以运行model.to('cuda:xxx'),不然会由于cos和sin是meta data而报错
        self.setup_rope_cache(max_length=self.max_length)
        return super().load_state_dict(state_dict, strict, assign)

    def model_size(self, include_embeddings: bool = True) -> int:
        """Calculate the model size.

        Args:
            include_embeddings (bool, optional): Include embeddings in the model size. Defaults to True.

        Returns:
            int: Model size
        """
        import itertools

        model_size = 0
        for n, children in self.named_children():
            if n == "embedding" and not include_embeddings:
                continue
            model_size += sum(
                [p.numel() * p.dtype.itemsize for p in itertools.chain(children.parameters(), children.buffers())]
            )
        return model_size


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    return cos, sin
