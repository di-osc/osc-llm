from abc import ABC, abstractmethod
import torch.nn as nn


class Quantizer(ABC):
    @abstractmethod
    def quantize(self, model: nn.Module) -> nn.Module:
        """在线量化模型,返回量化后的模型"""
        raise NotImplementedError

    @abstractmethod
    def convert_for_runtime(self, model: nn.Module) -> nn.Module:
        """仅将模型的结构转化为量化后的结构,并不对数据进行量化。

        Args:
            model (nn.Module): 待转换模型,不需要加载模型状态字典
        """
        raise NotImplementedError
