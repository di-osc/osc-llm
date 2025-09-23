from typing import Dict
import torch
from confection import Config

from ..registry import Registry
from .base import CausalLM


@Registry.models.register("Qwen3ForCausalLM")
class Qwen3ForCausalLM(CausalLM):
    hf_architecture = "Qwen3ForCausalLM"

    @property
    def weight_map(self) -> Dict:
        """获取qwen3权重映射表"""
        weight_map = {
            "model.embed_tokens.weight": "embedding.embed.weight",
            "model.norm.weight": "head_norm.weight",
            "lm_head.weight": "head.predictor.weight",
        }

        for i in range(self.hf_config["num_hidden_layers"]):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = (
                f"layers.{i}.attention_norm.weight"
            )
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = (
                f"layers.{i}.feedforward_norm.weight"
            )
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = (
                f"layers.{i}.attention.q_proj.weight"
            )
            weight_map[f"model.layers.{i}.self_attn.q_norm.weight"] = (
                f"layers.{i}.attention.q_norm.weight"
            )
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = (
                f"layers.{i}.attention.k_proj.weight"
            )
            weight_map[f"model.layers.{i}.self_attn.k_norm.weight"] = (
                f"layers.{i}.attention.k_norm.weight"
            )
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = (
                f"layers.{i}.attention.v_proj.weight"
            )
            weight_map[f"model.layers.{i}.self_attn.o_proj.weight"] = (
                f"layers.{i}.attention.o_proj.weight"
            )
            weight_map[f"model.layers.{i}.mlp.gate_proj.weight"] = (
                f"layers.{i}.feedforward.gate_proj.weight"
            )
            weight_map[f"model.layers.{i}.mlp.up_proj.weight"] = (
                f"layers.{i}.feedforward.up_proj.weight"
            )
            weight_map[f"model.layers.{i}.mlp.down_proj.weight"] = (
                f"layers.{i}.feedforward.down_proj.weight"
            )

        return weight_map

    @property
    def osc_config(self) -> Config:
        tempelate = """
        [model]
        @architecture = "TransformerDecoder"
        num_layers = {num_hidden_layers}
        prenorm = "True"

        [model.attention]
        @attention = "PagedAttention"
        in_dim = {hidden_size}
        num_heads = {num_attention_heads}
        head_dim = {head_dim}
        num_query_groups = {num_key_value_heads}
        rope_base = {rope_theta}
        q_bias = "False"
        k_bias = "False"
        v_bias = "False"
        o_bias = "False"
        
        [model.attention.q_norm]
        @normalization = "RMSNorm"
        in_dim = {head_dim}
        eps = {rms_norm_eps}

        [model.attention.k_norm]
        @normalization = "RMSNorm"
        in_dim = {head_dim}
        eps = {rms_norm_eps}

        [model.embedding]
        @embedding = "VocabEmbedding"
        num_embeddings = {vocab_size}
        embedding_dim = {hidden_size}

        [model.feedforward]
        @feedforward = "SwiGLU"
        in_dim = {hidden_size}
        hidden_dim = {intermediate_size}
        up_bias = "False"
        gate_bias = "False"
        down_bias = "False"

        [model.head]
        @head = "LMHead"
        in_dim = {hidden_size}
        out_dim = {vocab_size}
        bias = "False"

        [model.norm]
        @normalization = "RMSNorm"
        in_dim = {hidden_size}
        eps = {rms_norm_eps}
        """
        self.hf_config["max_length"] = self.hf_config.get(
            "max_length", self.hf_config["max_position_embeddings"]
        )
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)

    def load_checkpoint(
        self, model: torch.nn.Module, states: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        if "tie_word_embeddings" in self.hf_config:
            if self.hf_config["tie_word_embeddings"]:
                states["head.predictor.weight"] = states["embedding.embed.weight"]
        model.load_state_dict(states, strict=True)
        return model.eval()
