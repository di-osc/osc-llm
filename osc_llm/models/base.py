import json
from pathlib import Path
from typing import Dict, Union, Tuple

import torch
import torch.nn as nn
from wasabi import msg

from ..config import Config, registry
from ..tokenizer import Tokenizer
from ..chat_templates import ChatTemplate
from ..quantizers import Quantizer


class HFModel:
    """huggingface模型转换工具基类,一般情况下只需要完成`weight_map`属性和`osc_config`属性即可。"""

    hf_architecture: str

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        with open(self.checkpoint_dir / "config.json", "r") as f:
            self.hf_config = json.load(f)
        assert (
            self.hf_architecture in self.hf_config["architectures"]
        ), f"Only support {self.hf_architecture} model, current model is {self.hf_config['architectures']}"

    @property
    def weight_map(self) -> Dict[str, str]:
        """用来进行参数名称转换"""
        raise NotImplementedError("Method not implemented")

    @property
    def osc_config(self) -> Config:
        """用来构建osc格式模型的配置文件"""
        raise NotImplementedError("Method not implemented")

    def convert_checkpoint(self) -> Dict[str, torch.Tensor]:
        """将huggingface模型转换为osc格式模型

        Args:
            save_dir (str): 保存目录
        """
        pytorch_model = Path(self.checkpoint_dir) / "pytorch_model.bin"
        pytorch_idx_file = Path(self.checkpoint_dir) / "pytorch_model.bin.index.json"
        safetensors_model = Path(self.checkpoint_dir) / "model.safetensors"
        safetensors_idx_file = (
            Path(self.checkpoint_dir) / "model.safetensors.index.json"
        )
        if pytorch_model.exists() or pytorch_idx_file.exists():
            sd = self.convert_pytorch_format()
        elif safetensors_model.exists() or safetensors_idx_file.exists():
            sd = self.convert_safetensor_format()
        if (
            not pytorch_model.exists()
            and not safetensors_model.exists()
            and not pytorch_idx_file.exists()
            and not safetensors_idx_file.exists()
        ):
            raise FileNotFoundError("No pytorch_model.bin or model.safetensors found")
        return sd

    def convert_pytorch_format(self):
        sd = {}
        wmap = self.weight_map
        index_file = self.checkpoint_dir / "pytorch_model.bin.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
            files = [
                self.checkpoint_dir / file for file in set(index["weight_map"].values())
            ]
        else:
            files = [self.checkpoint_dir / "pytorch_model.bin"]
        assert len(files) > 0, "No pytorch model file found"
        for file in files:
            weights = torch.load(
                str(file), map_location="cpu", weights_only=True, mmap=True
            )
            for key in weights:
                if key not in wmap:
                    msg.warn(f"{key} not in wmap")
                    continue
                sd[wmap[key]] = weights[key]
        return sd

    def convert_safetensor_format(self):
        sd = {}
        wmap = self.weight_map
        index_file = self.checkpoint_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index = json.load(f)
            files = [
                self.checkpoint_dir / file for file in set(index["weight_map"].values())
            ]
        else:
            files = [self.checkpoint_dir / "model.safetensors"]
        assert len(files) > 0, "No pytorch model file found"
        try:
            from safetensors import safe_open
        except Exception:
            raise ImportError(
                "Please install safetensors first, run `pip install safetensors`"
            )
        for file in files:
            with safe_open(file, framework="pt") as f:
                for key in f.keys():
                    if key not in wmap:
                        msg.warn(f"{key} not in wmap")
                        continue
                    sd[wmap[key]] = f.get_tensor(key)
        return sd

    def build_model(
        self,
        config: Union[Dict, str, Path, Config],
        model_section: str = "model",
        empty_init: bool = True,
        return_config: bool = False,
    ) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Config]]:
        """Build a model from a configuration.

        Args:
            config (Union[Dict, str, Path, Config]): the configuration to build the model from, can be a dictionary, a path to a file or a Config object.
            model_section (str, optional): the section to look for the model in the configuration. Defaults to 'model'.
            empty_init (bool, optional): whether to initialize the model with empty weights. Defaults to True.
            return_config (bool, optional): whether to return the configuration as well. Defaults to False.

        Returns:
            torch.nn.Module: the model built from the configuration.
        """
        if isinstance(config, (str, Path)):
            config = Config().from_disk(config)
        if isinstance(config, dict):
            config = Config(data=config)
        if empty_init:
            with torch.device("meta"):
                resolved = registry.resolve(config=config)
        else:
            resolved = registry.resolve(config=config)
        if model_section not in resolved:
            msg.fail(f"cannot find model section {model_section}")
        else:
            model = resolved[model_section]
        if return_config:
            return model, config
        return model

    def quantize_model(
        self, model: nn.Module, quantizer: str | Quantizer = "int8"
    ) -> nn.Module:
        if isinstance(quantizer, str):
            quantizer: Quantizer = registry.quantizers.get(quantizer)()
        else:
            quantizer: Quantizer = quantizer
        model = quantizer.convert_for_runtime(model=model)
        return model

    def load_checkpoint(
        self, model: nn.Module, states: Dict[str, torch.Tensor]
    ) -> nn.Module:
        model.load_state_dict(
            state_dict=states,
            assign=True,
        )
        return model.eval()

    def load(self, quantizer: str | Quantizer | None = None) -> nn.Module:
        model = self.build_model(config=self.osc_config, empty_init=True)
        states = self.convert_checkpoint()
        model = self.load_checkpoint(model=model, states=states)
        if quantizer is not None:
            model = self.quantize_model(model=model, quantizer=quantizer)
        return model

    def load_tokenizer(self) -> Tokenizer:
        chat_template = ChatTemplate.from_name(self.checkpoint_dir.stem)
        if chat_template is None:
            chat_template = ChatTemplate.from_name(self.hf_config["architectures"][0])
        assert chat_template is not None, "No chat template found"
        tokenizer = Tokenizer(self.checkpoint_dir, chat_template=chat_template)
        return tokenizer

    def get_chat_template_config(self) -> ChatTemplate:
        for k, v in registry.chat_templates.get_all().items():
            if k in self.hf_config["architectures"]:  # 简单通过名称匹配
                config_str = f"""
                [chat_template]
                @chat_templates = {k}"""
                config = Config().from_str(config_str)
                return config
        return None

    def get_default_presision(self) -> str:
        if "torch_dtype" in self.hf_config:
            torch_precision = self.hf_config["torch_dtype"]
            return to_fabric_precision.get(torch_precision)

        return get_default_supported_precision()


to_fabric_precision = {"bfloat16": "bf16-true"}


def get_default_supported_precision(training: bool = False) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning_fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (
        torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    ):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def get_supported_hf_models():
    """获取支持的huggingface模型架构"""
    hf_models = []
    for model in registry.models.get_all():
        hf_models.append(model)
    return hf_models


def load_hf_model(checkpoint_dir: str) -> HFModel:
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_name = config["architectures"][0]
    allowed_models = get_supported_hf_models()
    if model_name not in allowed_models:
        msg.fail(
            title="Model {model_name} is not supported.",
            text=f"Supported models are: {allowed_models}",
            exits=1,
        )
    model: HFModel = registry.models.get(model_name)(checkpoint_dir)
    return model
