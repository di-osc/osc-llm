import torch.nn as nn
import torch
from typing import Optional
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


@registry.layers.register("ChatTTSEmbedding")
class ChatTTSEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_audio_tokens: int = 626,
        num_text_tokens: int = 32000,
        num_vq=4,
    ):
        super().__init__()

        self.num_vq = num_vq
        self.num_audio_tokens = num_audio_tokens

        self.model_dim = hidden_size
        self.emb_code = nn.ModuleList(
            [nn.Embedding(num_audio_tokens, self.model_dim) for _ in range(num_vq)],
        )
        self.emb_text = nn.Embedding(num_text_tokens, self.model_dim)

    def __call__(
        self, input_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> torch.Tensor:
        return super().__call__(input_ids, text_mask)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        emb_text: torch.Tensor = self.emb_text(
            input_ids[text_mask].narrow(1, 0, 1).squeeze_(1).to(device)
        )

        text_mask_inv = text_mask.logical_not().to(device)
        masked_input_ids: torch.Tensor = input_ids[text_mask_inv].to(device)

        emb_code = [
            self.emb_code[i](masked_input_ids[:, i]) for i in range(self.num_vq)
        ]
        emb_code = torch.stack(emb_code, 2).sum(2)

        emb = torch.zeros(
            (input_ids.shape[:-1]) + (emb_text.shape[-1],),
            device=emb_text.device,
            dtype=emb_text.dtype,
        )
        emb[text_mask] = emb_text
        emb[text_mask_inv] = emb_code.to(emb.dtype)

        del emb_text, emb_code, text_mask_inv

        return emb
