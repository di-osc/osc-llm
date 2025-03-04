from .llama import LlamaBuilder
from .qwen import Qwen2Builder, Qwen2MoeBuilder
from .chatglm import ChatGLM3Builder
from .base import HFModelBuilder
from ..config import registry

from pathlib import Path
import json
from wasabi import msg


def get_supported_architectures():
    """获取支持的huggingface模型架构"""
    architectures = []
    for model_builder in registry.model_builders.get_all():
        architectures.append(model_builder)
    return architectures


def get_hf_model_builder(checkpoint_dir: str) -> HFModelBuilder:
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    architecture = config["architectures"][0]
    allowed_architectures = get_supported_architectures()
    if architecture not in allowed_architectures:
        msg.fail(
            title="Architecture {architecture} is not supported.",
            text=f"Supported architectures are: {allowed_architectures}",
            exits=1,
        )
    model_builder: HFModelBuilder = registry.model_builders.get(architecture)(
        checkpoint_dir
    )
    return model_builder


__all__ = [
    "get_supported_architectures",
    "get_hf_model_builder",
    "HFModelBuilder",
    "LlamaBuilder",
    "Qwen2Builder",
    "Qwen2MoeBuilder",
    "ChatGLM3Builder",
]
