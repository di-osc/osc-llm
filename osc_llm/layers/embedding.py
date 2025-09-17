from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ..config import registry


@registry.layers.register("TokenEmbedding")
class TokenEmbedding(nn.Module):
    def __init__(self, n_embeddings: int, embedding_size: int):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=n_embeddings, embedding_dim=embedding_size
        )

    def forward(self, x, **kwargs):
        return self.embed(x)


@registry.layers.register("TokenEmbeddingPlus")
class TokenEmbeddingPlus(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        embedding_size: int,
        n_types: Optional[int] = None,
        n_positions: Optional[int] = None,
        norm: Optional[nn.Module] = None,
        dropout: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=n_embeddings, embedding_dim=embedding_size
        )
        if n_types:
            self.type_embed = nn.Embedding(
                num_embeddings=n_types, embedding_dim=embedding_size
            )
        if n_positions:
            self.pos_embed = nn.Embedding(
                num_embeddings=n_positions, embedding_dim=embedding_size
            )
            self.max_positions = n_positions
        if norm:
            self.norm = norm
        if dropout:
            self.dropout = dropout

    def forward(self, input_ids: torch.LongTensor, **kwargs):
        x = self.embed(input_ids)
        if hasattr(self, "type_embed"):
            token_type_ids = kwargs.get(
                "token_type_ids",
                torch.zeros_like(input_ids, device=x.device, dtype=torch.long),
            )
            x = x + self.type_embed(token_type_ids)
        if hasattr(self, "pos_embed"):
            B, L, D = x.size()
            if L > self.max_positions:
                raise ValueError(
                    f"Input length {L} is greater than the maximum number of positions {self.max_positions}."
                )
            input_pos = kwargs.get(
                "input_pos", torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            )
            x = x + self.pos_embed(input_pos)
        if hasattr(self, "norm"):
            x = self.norm(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x


@registry.layers.register("VocabParallelEmbedding")
class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
