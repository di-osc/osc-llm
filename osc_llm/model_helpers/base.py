import json
from pathlib import Path
from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
from ..config import Config, registry
from ..tokenizer import Tokenizer
from ..chat_templates import ChatTemplate
from wasabi import msg


class HFModelHelper:
    """huggingface模型转换工具基类,一般情况下只需要完成`weight_map`属性和`osc_config`属性即可。"""

    hf_architecture: str
    checkpoint_name: str = "osc_model.pth"

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        with open(self.checkpoint_dir / "config.json", "r") as f:
            self.hf_config = json.load(f)
        assert (
            self.hf_architecture in self.hf_config["architectures"]
        ), f'Only support {self.hf_architecture} model, current model is {self.hf_config["architectures"]}'
        try:
            self.tokenizer = Tokenizer(self.checkpoint_dir)
        except Exception:
            msg.warn("No tokenizer found")
            self.tokenizer = None

    @property
    def weight_map(self) -> Dict[str, str]:
        """用来进行参数名称转换"""
        raise NotImplementedError("Method not implemented")

    @property
    def osc_config(self) -> Config:
        """用来构建osc格式模型的配置文件"""
        raise NotImplementedError("Method not implemented")

    def convert_checkpoint(
        self,
        save_dir: str,
        add_chat_template: bool = True,
        save_new_states: bool = False,
    ) -> Dict | None:
        """将huggingface模型转换为osc格式模型

        Args:
            save_dir (str): 保存目录
        """
        pytorch_model = Path(self.checkpoint_dir) / "pytorch_model.bin"
        pytorch_idx_file = Path(self.checkpoint_dir) / "pytorch_model.bin.index.json"
        if pytorch_model.exists() or pytorch_idx_file.exists():
            sd = self.convert_pytorch_format()
        safetensors_model = Path(self.checkpoint_dir) / "model.safetensors"
        safetensors_idx_file = (
            Path(self.checkpoint_dir) / "model.safetensors.index.json"
        )
        if safetensors_model.exists() or safetensors_idx_file.exists():
            sd = self.convert_safetensor_format()
        if (
            not pytorch_model.exists()
            and not safetensors_model.exists()
            and not pytorch_idx_file.exists()
            and not safetensors_idx_file.exists()
        ):
            raise FileNotFoundError("No pytorch_model.bin or model.safetensors found")
        if save_new_states:
            out_dir = Path(save_dir)
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
            torch.save(sd, out_dir / "osc_model.pth")
            if add_chat_template:
                template_config = self.get_chat_template_config()
                if template_config:
                    config = self.osc_config.merge(template_config)
                else:
                    msg.warn("No chat template found")
            config = Config(data=config, section_order=["model", "chat_template"])
            config.to_disk(out_dir / "config.cfg")
            if self.tokenizer:
                self.tokenizer.save(out_dir)
        else:
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

    def load_checkpoint(
        self, model: nn.Module, states: Dict[str, torch.Tensor]
    ) -> nn.Module:
        model.load_state_dict(
            state_dict=states,
            assign=True,
        )
        return model.eval()

    def load_model(self) -> nn.Module:
        model = build_model(config=self.osc_config, empty_init=True)
        states = self.convert_checkpoint(self.checkpoint_dir, save_new_states=False)
        model = self.load_checkpoint(model=model, states=states)
        return model

    def get_chat_template_config(self) -> ChatTemplate:
        for k, v in registry.chat_templates.get_all().items():
            if k in self.checkpoint_dir.name:  # 简单通过名称匹配
                config_str = f"""
                [chat_template]
                @chat_templates = {k}"""
                config = Config().from_str(config_str)
                return config
        return None


def build_model(
    config: Union[Dict, str, Path, Config],
    model_section: str = "model",
    quantizer_section: str = "quantizer",
    empty_init: bool = True,
    quantize: bool = True,
    return_config: bool = False,
) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Config]]:
    """Build a model from a configuration.

    Args:
        config (Union[Dict, str, Path, Config]): the configuration to build the model from, can be a dictionary, a path to a file or a Config object.
        model_section (str, optional): the section to look for the model in the configuration. Defaults to 'model'.
        quantizer_section (str, optional): the section to look for the quantizer in the configuration. Defaults to 'quantizer'.
        empty_init (bool, optional): whether to initialize the model with empty weights. Defaults to True.
        quantize (bool, optional): whether to quantize the model. Defaults to True.
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
    if quantizer_section in resolved and quantize:
        quantizer = resolved[quantizer_section]
        model = quantizer.convert_for_runtime(model=model)
    if return_config:
        return model, config
    return model
