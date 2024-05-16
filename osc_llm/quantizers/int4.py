import torch
import torch.nn as nn
from typing import Union, Literal
from pathlib import Path
from ..layers.linear import WeightOnlyInt4Linear
from ..utils import find_multiple
from .base import Quantizer
from ..config import registry
from confection import Config


@registry.quantizers.register("WeightOnlyInt4Quantizer")
class WeightOnlyInt4Quantizer(Quantizer):
    def __init__(
        self,
        groupsize: Literal[32, 64, 128, 256] = 32,
        inner_k_tiles: Literal[2, 4, 8] = 8,
        padding_allowed: bool = True,
    ):
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    def quantize(self, model: nn.Module) -> nn.Module:
        for name, children in model.named_children():
            if isinstance(children, torch.nn.Linear):
                out_features = children.out_features
                in_features = children.in_features
                assert not children.bias
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                weight = children.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        print(f"warning: {name} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                        in_features = padded_in_features
                    else:
                        print(
                            f"warning: {name} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16), self.groupsize, self.inner_k_tiles
                )
                int4_linear = WeightOnlyInt4Linear(
                    in_features=in_features,
                    out_features=out_features,
                    groupsize=self.groupsize,
                    inner_k_tiles=self.inner_k_tiles,
                )
                int4_linear.weight = weight_int4pack
                int4_linear.scales_and_zeros = scales_and_zeros
                setattr(model, name, int4_linear)
            else:
                self.quantize(children)
        return model

    @torch.no_grad()
    def save_quantized_state_dict(self, model: nn.Module, save_path: Union[str, Path]):
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                weight = mod.weight.data
                if not _check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
                    weight.to(torch.bfloat16), self.groupsize, self.inner_k_tiles
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")
                if cur_state_dict.get(f"{fqn}.bias") is not None:
                    # remove bias
                    del cur_state_dict[f"{fqn}.bias"]

        save_path = Path(save_path)
        if save_path.exists():
            save_path.unlink()
        torch.save(cur_state_dict, save_path)

    def convert_for_runtime(self, model: nn.Module, use_cuda: bool = True):
        replace_linear_int4(model, self.groupsize, self.inner_k_tiles, self.padding_allowed, use_cuda)
        return model

    @property
    def quantizer_config(self) -> Config:
        config_str = f"""
        [quantizer]
        @quantizers = "WeightOnlyInt4Quantizer"
        groupsize = {self.groupsize}
        inner_k_tiles = {self.inner_k_tiles}
        padding_allowed = {self.padding_allowed}
        """
        config = Config().from_str(config_str)
        return config


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0


def replace_linear_int4(module, groupsize, inner_k_tiles, padding_allowed, use_cuda):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles) or padding_allowed:
                new_module = WeightOnlyInt4Linear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=False,
                    groupsize=groupsize,
                    inner_k_tiles=inner_k_tiles,
                )
                setattr(module, name, new_module)
        else:
            replace_linear_int4(child, groupsize, inner_k_tiles, padding_allowed, use_cuda)


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(weight_bf16, n_bit=4, groupsize=groupsize)
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros


def group_quantize_tensor(w: nn.Module, n_bit: int = 4, groupsize: int = 128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(torch.bfloat16).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    assert scales_and_zeros.dtype == torch.float
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int).to(torch.int32).reshape_as(w)
    return w_int32
