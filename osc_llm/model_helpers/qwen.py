from typing import Dict
from ..config import Config, registry
from .base import HFModelHelper


@registry.model_helpers.register("Qwen2ForCausalLM")
class Qwen2Helper(HFModelHelper):
    hf_architecture = "Qwen2ForCausalLM"

    @property
    def weight_map(self) -> Dict:
        """获取qwen2权重映射表"""
        weight_map = {
            "model.embed_tokens.weight": "embedding.embed.weight",
            "model.norm.weight": "head_norm.weight",
            "lm_head.weight": "head.weight",
        }

        for i in range(self.hf_config["num_hidden_layers"]):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.attention_norm.weight"
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.feedforward_norm.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attention.q_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.bias"] = f"blocks.{i}.attention.q_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attention.k_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.k_proj.bias"] = f"blocks.{i}.attention.k_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attention.v_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.v_proj.bias"] = f"blocks.{i}.attention.v_proj.bias"
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
        q_bias = "True"
        k_bias = "True"
        v_bias = "True"
        o_bias = "False"

        [model.embedding]
        @layers = "TokenEmbedding"
        n_embeddings = {vocab_size}
        embedding_size = {hidden_size}

        [model.feedforward]
        @layers = "SwiGLU"
        n_in = {hidden_size}
        n_hidden = {intermediate_size}
        up_bias = "False"
        gate_bias = "False"
        down_bias = "False"

        [model.head]
        @layers = "Linear"
        n_in = {hidden_size}
        n_out = {vocab_size}
        bias = "False"

        [model.norm]
        @layers = "RMSNorm"
        n_in = {hidden_size}
        eps = 0.000001
        """
        self.hf_config["max_length"] = self.hf_config.get("max_length", self.hf_config["max_position_embeddings"])
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)


@registry.model_helpers.register("Qwen2MoeForCausalLM")
class Qwen2MoeHelper(HFModelHelper):
    hf_architecture = "Qwen2MoeForCausalLM"

    @property
    def weight_map(self) -> Dict:
        """qwen2moe权重映射表"""
        weight_map = {
            "model.embed_tokens.weight": "embedding.embed.weight",
            "model.norm.weight": "head_norm.weight",
            "lm_head.weight": "head.weight",
        }

        for i in range(self.hf_config["num_hidden_layers"]):
            weight_map[f"model.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.attention_norm.weight"
            weight_map[f"model.layers.{i}.post_attention_layernorm.weight"] = f"blocks.{i}.feedforward_norm.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attention.q_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.q_proj.bias"] = f"blocks.{i}.attention.q_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attention.k_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.k_proj.bias"] = f"blocks.{i}.attention.k_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attention.v_proj.weight"
            weight_map[f"model.layers.{i}.self_attn.v_proj.bias"] = f"blocks.{i}.attention.v_proj.bias"
            weight_map[f"model.layers.{i}.self_attn.o_proj.weight"] = f"blocks.{i}.attention.o_proj.weight"
            weight_map[f"model.layers.{i}.mlp.gate.weight"] = f"blocks.{i}.feedforward.gate.weight"

            # gate
            weight_map[f"model.layers.{i}.mlp.gate.weight"] = f"blocks.{i}.feedforward.gate.weight"

            # shared expert
            weight_map[f"model.layers.{i}.mlp.shared_expert.up_proj.weight"] = (
                f"blocks.{i}.feedforward.shared_expert.up_proj.weight"
            )
            weight_map[f"model.layers.{i}.mlp.shared_expert.down_proj.weight"] = (
                f"blocks.{i}.feedforward.shared_expert.down_proj.weight"
            )
            weight_map[f"model.layers.{i}.mlp.shared_expert.gate_proj.weight"] = (
                f"blocks.{i}.feedforward.shared_expert.gate_proj.weight"
            )
            weight_map[f"model.layers.{i}.mlp.shared_expert_gate.weight"] = f"blocks.{i}.feedforward.shared_gate.weight"

            # experts
            for j in range(self.hf_config["num_experts"]):
                weight_map[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = (
                    f"blocks.{i}.feedforward.experts.{j}.up_proj.weight"
                )
                weight_map[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = (
                    f"blocks.{i}.feedforward.experts.{j}.down_proj.weight"
                )
                weight_map[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = (
                    f"blocks.{i}.feedforward.experts.{j}.gate_proj.weight"
                )

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
        q_bias = "True"
        k_bias = "True"
        v_bias = "True"
        o_bias = "False"

        [model.embedding]
        @layers = "TokenEmbedding"
        n_embeddings = {vocab_size}
        embedding_size = {hidden_size}

        [model.feedforward]
        @layers = "SparseMoe"
        n_experts = {num_experts}
        n_activated_experts = {num_experts_per_tok}
        norm_probs = {norm_topk_prob}
        
        [model.feedforward.expert]
        @layers = "SwiGLU"
        n_in = {hidden_size}
        n_hidden = {moe_intermediate_size}
        up_bias = "False"
        down_bias = "False"
        gate_bias = "False"
        
        [model.feedforward.gate]
        @layers = "Linear"
        n_in = {hidden_size}
        n_out = {num_experts}
        bias = "False"
        
        [model.feedforward.shared_expert]
        @layers = "SwiGLU"
        n_in = {hidden_size}
        n_hidden = {intermediate_size}
        up_bias = "False"
        down_bias = "False"
        gate_bias = "False"
        
        [model.feedforward.shared_gate]
        @layers = "Linear"
        n_in = {hidden_size}
        n_out = 1
        bias = "False"

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
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)
