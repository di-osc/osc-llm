import torch.nn as nn
from ..config import registry
from typing import Optional
import torch



@registry.layers.register("TokenEmbedding")
class TokenEmbedding(nn.Module):
    def __init__(self, 
                 n_embeddings: int,
                 embedding_size: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_size)
        
    def forward(self, x, **kwargs):
        return self.embed(x)


@registry.layers.register("TokenEmbeddingPlus")
class TokenEmbeddingPlus(nn.Module):
    def __init__(self, 
                 n_embeddings: int,
                 embedding_size: int,
                 n_types: Optional[int] = None,
                 n_positions: Optional[int] = None,
                 norm: Optional[nn.Module] = None,
                 dropout: Optional[nn.Module] = None,
                 ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=embedding_size)
        if n_types:
            self.type_embed = nn.Embedding(num_embeddings=n_types, embedding_dim=embedding_size)
        if n_positions:
            self.pos_embed = nn.Embedding(num_embeddings=n_positions, embedding_dim=embedding_size)
            self.max_positions = n_positions
        if norm:
            self.norm = norm
        if dropout:
            self.dropout = dropout
        
    def forward(self, input_ids: torch.LongTensor, **kwargs):
        x = self.embed(input_ids)
        if hasattr(self, "type_embed"):
            token_type_ids = kwargs.get("token_type_ids", torch.zeros_like(input_ids, device=x.device, dtype=torch.long))
            x = x + self.type_embed(token_type_ids)
        if hasattr(self, "pos_embed"):
            B, L, D = x.size()
            if L > self.max_positions:
                raise ValueError(f"Input length {L} is greater than the maximum number of positions {self.max_positions}.")
            input_pos = kwargs.get("input_pos", torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1))
            x = x + self.pos_embed(input_pos)
        if hasattr(self, "norm"):
            x = self.norm(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x