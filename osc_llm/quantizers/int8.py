from .base import Quantizer
from ..layers import Int8Linear
from ..config import registry
from confection import Config
import torch.nn as nn
import torch


@registry.quantizers.register("Int8Quantizer")
class Int8Quantizer(Quantizer):
    def quantize(self, model: nn.Module) -> nn.Module:
        for name, children in model.named_children():
            if isinstance(children, torch.nn.Linear):
                int8_weight, scales, _ = self._dynamically_quantize_per_channel(
                    children.weight.float(), -128, 127, torch.int8
                )
                if hasattr(children, "bias") and children.bias is not None:
                    int8_linear = Int8Linear(
                        in_features=children.in_features,
                        out_features=children.out_features,
                        bias=True,
                    )
                    int8_linear.bias = children.bias
                    int8_linear.weight = int8_weight
                    int8_linear.scales = scales
                else:
                    int8_linear = Int8Linear(
                        in_features=children.in_features,
                        out_features=children.out_features,
                    )
                    int8_linear.weight = int8_weight
                    int8_linear.scales = scales
                setattr(model, name, int8_linear)
            else:
                self.quantize(model=children)
        return model

    def convert_for_runtime(self, model: nn.Module) -> nn.Module:
        model = self._replace_linear_weight_only_int8_per_channel(model)
        return model

    @property
    def quantizer_config(self):
        config_str = """
        [quantizer]
        @quantizers = "Int8Quantizer"
        """
        config = Config().from_str(config_str)
        return config

    def _replace_linear_weight_only_int8_per_channel(self, module: nn.Module) -> nn.Module:
        """递归替换module中的所有nn.Linear为WeightOnlyInt8Linear"""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if hasattr(child, "bias") and child.bias is not None:
                    setattr(
                        module,
                        name,
                        Int8Linear(child.in_features, child.out_features, bias=True),
                    )
                else:
                    setattr(module, name, Int8Linear(child.in_features, child.out_features))
            else:
                self._replace_linear_weight_only_int8_per_channel(child)
        return module

    def _dynamically_quantize_per_channel(
        self, x, quant_min=-128, quant_max=127, target_dtype: torch.dtype = torch.int8
    ):
        # assumes symmetric quantization
        # assumes axis == 0
        # assumes dense memory format
        # TODO(future): relax ^ as needed

        # default setup for affine quantization of activations
        eps = torch.finfo(torch.float32).eps

        # get min and max
        min_val, max_val = torch.aminmax(x, dim=-1)

        # calculate scales and zero_points based on min and max
        # reference: https://fburl.com/code/srbiybme
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device

        # reference: https://fburl.com/code/4wll53rk
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scales = max_val_pos / (float(quant_max - quant_min) / 2)
        # ensure scales is the same dtype as the original tensor
        scales = torch.clamp(scales, min=eps).to(x.dtype)
        zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        # quantize based on qmin/qmax/scales/zp
        # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
        x_div = x / scales.unsqueeze(-1)
        x_round = torch.round(x_div)
        x_zp = x_round + zero_points.unsqueeze(-1)
        quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

        return quant, scales, zero_points
