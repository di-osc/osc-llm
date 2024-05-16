from .llama import LlamaHelper
from .qwen import Qwen2Helper
from .chatglm import ChatGLM3Helper


from ..config import registry


def get_supported_architectures():
    """获取支持的huggingface模型架构"""
    architectures = []
    for model_helper in registry.model_helpers.get_all():
        architectures.append(model_helper)
    return architectures


__all__ = [
    "LlamaHelper",
    "Qwen2Helper",
    "ChatGLM3Helper",
    "get_supported_architectures",
]
