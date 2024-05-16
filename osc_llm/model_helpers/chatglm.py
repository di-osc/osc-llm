from .base import HFModelHelper
from ..config import Config


class ChatGLM3Helper(HFModelHelper):
    hf_architecture: str = "ChatGLMModel"

    @property
    def weight_map(self):
        weight_map = {
            "transformer.embedding.word_embeddings.weight": "embedding.embed.weight",
            "transformer.encoder.final_layernorm.weight": "head_norm.weight",
            "transformer.output_layer.weight": "head.weight",
        }

        for i in range(self.hf_config["num_layers"]):
            weight_map[f"transformer.encoder.layers.{i}.input_layernorm.weight"] = f"blocks.{i}.attention_norm.weight"
            weight_map[f"transformer.encoder.layers.{i}.post_attention_layernorm.weight"] = (
                f"blocks.{i}.feedforward_norm.weight"
            )
            weight_map[f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight"] = (
                f"blocks.{i}.attention.qkv_proj.weight"
            )
            weight_map[f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias"] = (
                f"blocks.{i}.attention.qkv_proj.bias"
            )
            weight_map[f"transformer.encoder.layers.{i}.self_attention.dense.weight"] = (
                f"blocks.{i}.attention.o_proj.weight"
            )
            weight_map[f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight"] = (
                f"blocks.{i}.feedforward.up_gate_proj.weight"
            )
            weight_map[f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight"] = (
                f"blocks.{i}.feedforward.down_proj.weight"
            )

        return weight_map

    @property
    def osc_config(self) -> Config:
        tempelate = """
        [model]
        @architectures = "TransformerDecoder"
        n_blocks = {num_layers}
        block_size = {seq_length}
        prenorm = "True"
        rope_base = 10000

        [model.attention]
        @layers = "CausalSelfAttention"
        n_in = {hidden_size}
        n_heads = {num_attention_heads}
        n_query_groups = {multi_query_group_num}
        use_qkv_proj = "True"
        qkv_bias = {add_qkv_bias}
        o_bias = "False"

        [model.embedding]
        @layers = "TokenEmbedding"
        n_embeddings = {padded_vocab_size}
        embedding_size = {hidden_size}

        [model.feedforward]
        @layers = "SwiGLU.v2"
        n_in = {hidden_size}
        n_hidden = {ffn_hidden_size}
        up_gate_bias = "False"
        down_bias = "False"

        [model.head]
        @layers = "Linear"
        n_in = {hidden_size}
        n_out = {padded_vocab_size}
        bias = "False"

        [model.norm]
        @layers = "RMSNorm"
        n_in = {hidden_size}
        eps = {layernorm_epsilon}
        """
        config_str = tempelate.format(**self.hf_config)
        return Config().from_str(config_str)
