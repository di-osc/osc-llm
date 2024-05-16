from typing import Dict
from ..config import Config, registry
from .base import HFModelHelper


@registry.model_helpers.register("LlamaForCausalLM")
class LlamaHelper(HFModelHelper):
    hf_architecture = "LlamaForCausalLM"

    @property
    def weight_map(self) -> Dict:
        """获取llama2权重映射表"""
        weight_map = {
            "model.embed_tokens.weight": "embedding.embed.weight",
            "model.norm.weight": "head_norm.weight",
            "lm_head.weight": "head.weight",
        }

        for i in range(self.hf_config["num_hidden_layers"]):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.attention_norm.weight"
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.feedforward_norm.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attention.q_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attention.k_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attention.v_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.o_proj.weight"] = f"blocks.{i}.attention.o_proj.weight"
            weight_map[f"model.layers.{i}.mlp.gate_proj.weight"] = f"blocks.{i}.feedforward.gate_proj.weight"
            weight_map[f"model.layers.{i}.mlp.up_proj.weight"] = f"blocks.{i}.feedforward.up_proj.weight"
            weight_map[f"model.layers.{i}.mlp.down_proj.weight"] = f"blocks.{i}.feedforward.down_proj.weight"

        return weight_map

    @property
    def osc_config(self) -> Config:
        tempelate = """
        [model]
        @architectures = "TransformerDecoder"
        n_blocks = {num_hidden_layers}
        block_size = {max_length}
        prenorm = "True"
        rope_base = {rope_theta}

        [model.attention]
        @layers = "CausalSelfAttention"
        n_in = {hidden_size}
        n_heads = {num_attention_heads}
        n_query_groups = {num_key_value_heads}
        q_bias = "False"
        k_bias = "False"
        v_bias = "False"
        o_bias = "False"

        [model.embedding]
        @layers = "TokenEmbedding"
        n_embeddings = {vocab_size}
        embedding_size = {hidden_size}

        [model.feedforward]
        @layers = "SwiGLU"
        n_in = {hidden_size}
        n_hidden = {intermediate_size}

        [model.head]
        @layers = "Linear"
        n_in = {hidden_size}
        n_out = {vocab_size}
        bias = "False"

        [model.norm]
        @layers = "RMSNorm"
        n_in = {hidden_size}
        eps = {rms_norm_eps}
        """
        self.hf_config["max_length"] = self.hf_config.get("max_length", self.hf_config["max_position_embeddings"])
        self.hf_config["rope_theta"] = self.hf_config.get("rope_theta", 10000)
        self.hf_config["rms_norm_eps"] = self.hf_config.get("rms_norm_eps", 1e-5)
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)
