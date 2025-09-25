import json
from pathlib import Path
from typing import Dict, Union, Tuple, List, Generator

import torch
import torch.nn as nn
from loguru import logger
from osc_transformers import TransformerDecoder, SamplingParams, Sequence
from confection import Config

from ..registry import Registry
from ..tokenizer import Tokenizer
from ..chat_templates import ChatTemplate


class CausalLM:
    """huggingface因果语言模型基类,一般情况下只需要完成`weight_map`属性和`osc_config`属性即可。"""

    hf_architecture: str

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        with open(self.checkpoint_dir / "config.json", "r") as f:
            self.hf_config: Dict = json.load(f)
        assert self.hf_architecture in self.hf_config["architectures"], (
            f"Only support {self.hf_architecture} model, current model is {self.hf_config['architectures']}"
        )
        self.tokenizer = Tokenizer(checkpoint_dir=self.checkpoint_dir)
        self.model: TransformerDecoder = self.load()

    def setup(self, gpu_memory_utilization: float = 0.5, device: str = "cuda"):
        max_model_len = self.hf_config.get("max_length", 4096)
        dtype = self.hf_config.get("torch_dtype", "bfloat16")
        dtype = str_to_dtype(dtype)
        self.model.setup(
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            eos=self.tokenizer.eos_id,
            dtype=dtype,
            device=device,
            model_name=self.hf_architecture,
        )

    def stream(
        self, prompt: str, sampling_params: SamplingParams | None = None
    ) -> Generator[str, None, None]:
        token_ids = self.tokenizer.encode(string=prompt).tolist()
        if sampling_params is None:
            sampling_params = SamplingParams()
        seq = Sequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
        )
        return self.tokenizer.decode_stream(self.model.stream(seq=seq))

    def generate(
        self,
        prompts: List[str] | str,
        sampling_params: List[SamplingParams] | None = None,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_token_ids = [
            self.tokenizer.encode(string=prompt).tolist() for prompt in prompts
        ]
        if sampling_params is None:
            sampling_params = [SamplingParams() for _ in prompts]
        seqs = [
            Sequence(token_ids=token_ids, sampling_params=sampling_params)
            for token_ids, sampling_params in zip(batch_token_ids, sampling_params)
        ]
        seqs = self.model.batch(seqs=seqs)
        results = []
        for seq in seqs:
            content = self.tokenizer.decode(seq.completion_token_ids)
            results.append(content)
        return results

    @property
    def weight_map(self) -> Dict[str, str]:
        """用来进行参数名称转换"""
        raise NotImplementedError("Method not implemented")

    @property
    def osc_config(self) -> Config:
        """用来构建osc格式模型的配置文件"""
        raise NotImplementedError("Method not implemented")

    def get_chat_template(self) -> ChatTemplate:
        return ChatTemplate.from_hf_architecture(self.hf_architecture)

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
                    logger.warning(f"{key} not in wmap")
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
                        logger.warning(f"{key} not in wmap")
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
        model = TransformerDecoder.from_config(
            config, empty_init=empty_init, model_section=model_section
        )
        if return_config:
            return model, config
        return model

    def load_checkpoint(
        self, model: nn.Module, states: Dict[str, torch.Tensor]
    ) -> nn.Module:
        model.load_state_dict(
            state_dict=states,
            assign=True,
        )
        return model.eval()

    def load(self) -> nn.Module:
        model = self.build_model(config=self.osc_config, empty_init=True)
        states = self.convert_checkpoint()
        model = self.load_checkpoint(model=model, states=states)
        return model

    def get_default_presision(self) -> str:
        if "torch_dtype" in self.hf_config:
            torch_precision = str_to_dtype(self.hf_config["torch_dtype"])
            return torch_precision
        return torch.get_default_dtype()


def str_to_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_supported_hf_models():
    """获取支持的huggingface模型架构"""
    hf_models = []
    for model in Registry.models.get_all():
        hf_models.append(model)
    return hf_models


def load_causal_lm(checkpoint_dir: str) -> CausalLM:
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_name = config["architectures"][0]
    allowed_models = get_supported_hf_models()
    if model_name not in allowed_models:
        logger.error(
            f"Model {model_name} is not supported. Supported models are: {allowed_models}"
        )
    model: CausalLM = Registry.models.get(model_name)(checkpoint_dir)
    return model
