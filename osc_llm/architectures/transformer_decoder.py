from typing import Mapping, Optional, List, Any
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
        input_pos: Optional[torch.Tensor] = None,
    ):
        if self.prenorm:
            x = (
                self.attention(
                    self.attention_norm(x),
                    input_pos=input_pos,
                )
                + x
            )
            x = x + self.feedforward(self.feedforward_norm(x))
        else:
            x = self.attention_norm(
                self.attention(
                    x,
                    input_pos=input_pos,
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
        self.attn_ctx: AttentionContext = AttentionContext()
        for layer in self.layers:
            layer.attention.setup(self.attn_ctx, init_kv_cache=False)

    def forward(
        self, input_ids: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ):
        """Forward pass of the TransformerDecoder.

        Args:
            input_ids (torch.Tensor): Input token ids. shape = (seq_length)
            input_pos (Optional[torch.Tensor], optional): Input position ids. prefill stage shape = (seq_length) decode stage shape = (batch_size, 1). Defaults to None.
        """
        assert len(input_ids.shape) == 1, "input must be 1d"
        L = input_ids.size()[0]

        if self.max_length < L:
            raise ValueError(
                f"Cannot forward sequence of length {L}, max seq length is only {self.max_length}."
            )

        if input_pos is None:
            input_pos = torch.arange(L, dtype=torch.int32)

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, input_pos=input_pos)

        if self.prenorm:
            x = self.head_norm(x)

        return x

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_ctx.is_prefill:
            last_indices = self.attn_ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
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
            int: Model size
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
        return model_size

    def set_kv_cache(self, num_kvcache_blocks: int, block_size: int) -> None:
        self.attn_ctx.num_kvcache_blocks = num_kvcache_blocks
        self.attn_ctx.block_size = block_size
        for layer in self.layers:
            layer.attention.setup(self.attn_ctx, init_kv_cache=True)

    def set_attn_ctx(
        self,
        is_prefill: bool = False,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        slot_mapping: torch.Tensor = None,
        context_lens: torch.Tensor = None,
        block_tables: torch.Tensor = None,
    ):
        self.attn_ctx.is_prefill = is_prefill
        self.attn_ctx.cu_seqlens_q = cu_seqlens_q
        self.attn_ctx.cu_seqlens_k = cu_seqlens_k
        self.attn_ctx.max_seqlen_q = max_seqlen_q
        self.attn_ctx.max_seqlen_k = max_seqlen_k
        self.attn_ctx.slot_mapping = slot_mapping
        self.attn_ctx.context_lens = context_lens
        self.attn_ctx.block_tables = block_tables

    def reset_attn_ctx(self):
        self.attn_ctx.reset_run_info()

    def prepare_prefill(
        self,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        block_tables,
    ):
        self.attn_ctx.is_prefill = True
        self.attn_ctx.cu_seqlens_q = cu_seqlens_q
        self.attn_ctx.cu_seqlens_k = cu_seqlens_k
        self.attn_ctx.max_seqlen_q = max_seqlen_q
        self.attn_ctx.max_seqlen_k = max_seqlen_k
        self.attn_ctx.slot_mapping = slot_mapping
        self.attn_ctx.context_lens = None
        self.attn_ctx.block_tables = block_tables

    def prepare_decode(self, context_lens, block_tables, slot_mapping):
        self.attn_ctx.is_prefill = False
        self.attn_ctx.context_lens = context_lens
        self.attn_ctx.block_tables = block_tables
        self.attn_ctx.slot_mapping = slot_mapping
