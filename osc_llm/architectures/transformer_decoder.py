from typing import Mapping, List, Any
from copy import deepcopy

import torch
import torch.nn as nn

from ..config import registry
from ..layers.attention import Attention, AttentionContext


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        attention: Attention,
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
        attn_ctx: AttentionContext,
    ):
        if self.prenorm:
            x = (
                self.attention(
                    self.attention_norm(x),
                    attn_ctx=attn_ctx,
                )
                + x
            )
            x = x + self.feedforward(self.feedforward_norm(x))
        else:
            x = self.attention_norm(
                self.attention(
                    x,
                    attn_ctx=attn_ctx,
                )
                + x
            )
            x = self.feedforward_norm(self.feedforward(x) + x)
        return x


@registry.architectures.register("TransformerDecoder")
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        max_length: int,
        embedding: nn.Module,
        attention: Attention,
        feedforward: nn.Module,
        head: nn.Module,
        norm: nn.Module,
        prenorm: bool = True,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.n_layers = n_layers
        self.embedding = embedding
        self.layers: List[TransformerDecoderLayer] = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    attention=deepcopy(attention),
                    attention_norm=deepcopy(norm),
                    feedforward=deepcopy(feedforward),
                    feedforward_norm=deepcopy(norm),
                    prenorm=prenorm,
                )
                for _ in range(n_layers)
            ]
        )
        self.head_norm = norm if self.prenorm else None
        self.head = head

        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_ctx: AttentionContext,
    ):
        """Forward pass of the TransformerDecoder.

        Args:
            input_ids (torch.Tensor): Input token ids. shape = (seq_length)
            attn_ctx (AttentionContext): Attention context.
            input_pos (Optional[torch.Tensor], optional): Input position ids. prefill stage shape = (seq_length) decode stage shape = (batch_size, 1). Defaults to None.
        """
        assert len(input_ids.shape) == 1, "input must be 1d"
        L = input_ids.size()[0]

        if self.max_length < L:
            raise ValueError(
                f"Cannot forward sequence of length {L}, max seq length is only {self.max_length}."
            )

        if attn_ctx.input_pos is None:
            attn_ctx.input_pos = torch.arange(L, dtype=torch.int32)

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, attn_ctx=attn_ctx)

        if self.prenorm:
            x = self.head_norm(x)

        if attn_ctx.is_prefill:
            last_indices = attn_ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        return x

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True
    ):
        # 保证在用torch.device('meta')构建模型后, 可以运行model.to('cuda:xxx'),不然会由于cos和sin是meta data而报错
        return super().load_state_dict(state_dict, strict, assign)

    def model_size(self, include_embeddings: bool = True) -> int:
        """Calculate the model size.

        Args:
            include_embeddings (bool, optional): Include embeddings in the model size. Defaults to True.

        Returns:
            int: Model size in MB
        """
        import itertools

        model_size = 0
        for n, children in self.named_children():
            if n == "embedding" and not include_embeddings:
                continue
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(children.parameters(), children.buffers())
                ]
            )
        return model_size / 1024 / 1024

    def set_kv_cache(self, num_kvcache_blocks: int, block_size: int) -> None:
        for layer in self.layers:
            layer.attention.set_kv_cache(num_kvcache_blocks, block_size)
