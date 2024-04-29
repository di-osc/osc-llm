from typing import Union, List, Iterator, Optional
from lightning import Fabric
from time import perf_counter
import sys
from ..samplers import Sampler, TopK


class LLMEngine:
    """语言模型引擎: 控制着大模型加载,编译,运转以及停止。
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        sampler: Optional[Sampler] = None,
        max_length: Optional[int] = None,
        devices: Union[int, List[int]] = 1,
        accelerator: str = "auto",
        compile: bool = True,
        warmup: bool = False
    ):

        self.fabric = Fabric(devices=devices, accelerator=accelerator, precision="bf16-true")
        
        self.sampler = sampler if sampler else TopK(temperature=0.8, k=200)
        self.max_length = max_length
        
        self.compile = compile
        self.warmup = warmup
        
        self.checkpoint_dir = checkpoint_dir
    
    def load_model(self) -> None:
        raise NotImplementedError
    
    def compile_model(self) -> None:
        raise NotImplementedError
    
    def warmup_model(self) -> None:
        raise NotImplementedError
    
    def run(self, **model_inputs) -> Iterator[str]:
        raise NotImplementedError
    
    def setup(self) -> None:
        t = perf_counter()
        self.load_model()
        self.fabric.print(f"load model in {perf_counter() - t:.02f} seconds", file=sys.stderr)
        if self.compile:
            t = perf_counter()
            self.compile_model()
            self.fabric.print(f"compile model in {perf_counter() - t:.02f} seconds", file=sys.stderr)
        if self.warmup:
            t = perf_counter()
            self.warmup_model()
            self.fabric.print(f"warmup model in {perf_counter() - t:.02f} seconds", file=sys.stderr)
            
    def reset_sampler(self, sampler: Sampler) -> None:
        self.sampler = sampler