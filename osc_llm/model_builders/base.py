import json
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from ..config import Config, registry
from ..tokenizer import Tokenizer
from ..chat_templates import ChatTemplate
from ..utils import build_model
from wasabi import msg


class HFModelBuilder:
    """huggingface模型转换工具基类,一般情况下只需要完成`weight_map`属性和`osc_config`属性即可。"""

    hf_architecture: str

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        with open(self.checkpoint_dir / "config.json", "r") as f:
            self.hf_config = json.load(f)
        assert self.hf_architecture in self.hf_config["architectures"], (
            f"Only support {self.hf_architecture} model, current model is {self.hf_config['architectures']}"
        )

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
        states = self.convert_checkpoint()
        model = self.load_checkpoint(model=model, states=states)
        return model

    def load_tokenizer(self) -> Tokenizer:
        chat_template = ChatTemplate.from_name(self.checkpoint_dir.stem)
        if chat_template is None:
            chat_template = ChatTemplate.from_name(self.hf_config["architectures"][0])
        print(chat_template)
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
