from typing import Union, List, Optional, Generator
from lightning import Fabric
from time import perf_counter
import sys
from ..samplers import Sampler, TopK
from abc import ABC, abstractmethod
import torch
from ..utils import get_default_supported_precision


class LLMEngine(ABC):
    """语言模型引擎: 控制着大模型加载,编译,运转以及停止。"""

    def __init__(
        self,
        checkpoint_dir: str,
        draft_checkpoint_dir: Optional[str] = None,
        sampler: Optional[Sampler] = None,
        max_length: Optional[int] = None,
        devices: Union[int, List[int]] = 1,
        accelerator: str = "auto",
        compile: bool = True,
        precision: Optional[str] = None,
    ):
        if not precision:
            precision = get_default_supported_precision(training=False)
        self.fabric = Fabric(devices=devices, accelerator=accelerator, precision=precision)

        self.sampler = sampler if sampler else TopK(temperature=0.8, k=200)
        self.max_length = max_length

        self.compile = compile

        self.checkpoint_dir = checkpoint_dir
        self.draft_checkpoint_dir = draft_checkpoint_dir

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    def compile_model(self) -> None:
        raise NotImplementedError

    def setup_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, **model_inputs) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError

    def setup(self) -> None:
        t = perf_counter()
        self.load_model()
        self.fabric.print(f"load model in {perf_counter() - t:.02f} seconds", file=sys.stderr)
        if self.compile:
            self.compile_model()
        t = perf_counter()
        self.setup_model()
        self.fabric.print(f"setup model in {perf_counter() - t:.02f} seconds", file=sys.stderr)

    def reset_sampler(self, sampler: Sampler) -> None:
        self.sampler = sampler
