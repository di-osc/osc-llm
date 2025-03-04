from osc_llm.samplers import Sampler, TopK
from osc_llm.utils import get_default_supported_precision, get_hf_model_helper
from osc_llm.architectures import TransformerDecoder
from osc_llm.tokenizer import Tokenizer, Message
from typing import Union, List, Optional, Generator, Iterable
from torch.nn.attention import sdpa_kernel, SDPBackend
from lightning import Fabric
from time import perf_counter
import torch
import sys

torch.set_float32_matmul_precision("high")


class LLM:
    """语言模型引擎: 控制着大模型加载,编译,运转以及停止。"""

    def __init__(
        self,
        checkpoint_dir: str,
        draft_checkpoint_dir: Optional[str] = None,
        speculate_k: int = 8,
        sampler: Optional[Sampler] = None,
        max_length: Optional[int] = None,
        devices: Union[int, List[int]] = 1,
        accelerator: str = "auto",
        precision: Optional[str] = None,
        compile: bool = False,
        compile_prefill: bool = False,
    ):
        if not precision:
            precision = get_default_supported_precision(training=False)
        self.fabric = Fabric(
            devices=devices, accelerator=accelerator, precision=precision
        )
        self.tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
        self.sampler = sampler if sampler else TopK(temperature=0.8, k=200)
        self.max_length = max_length

        self.checkpoint_dir = checkpoint_dir
        self.model: TransformerDecoder = None
        self.draft_checkpoint_dir = draft_checkpoint_dir
        self.speculate_k = speculate_k

        self.setup(compile=compile, compile_prefille=compile_prefill)

    def setup(self, compile: bool = False, compile_prefille: bool = False) -> None:
        t = perf_counter()
        self.load_model()
        self.fabric.print(
            f"load model in {perf_counter() - t:.02f} seconds", file=sys.stderr
        )
        if compile:
            self.compile_model(compile_prefill=compile_prefille)
        t = perf_counter()
        self.setup_model()
        self.fabric.print(
            f"setup model in {perf_counter() - t:.02f} seconds", file=sys.stderr
        )

    def load_model(self) -> None:
        model_helper = get_hf_model_helper(self.checkpoint_dir)
        self.model: TransformerDecoder = model_helper.load_model()
        self.model = self.fabric.to_device(self.model)

        with self.fabric.init_tensor():
            self.model.setup_kv_cache(
                batch_size=1, max_length=self.max_length, dtype=torch.bfloat16
            )

    def setup_model(self) -> None:
        self.model = self.fabric.setup_module(self.model, move_to_device=True)

    def compile_model(self, compile_prefill: bool = False) -> None:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
        torch._inductor.config.triton.cudagraph_trees = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        self.decode = torch.compile(self.decode, fullgraph=True, mode="reduce-overhead")
        if compile_prefill:
            self.prefill = torch.compile(self.prefill, dynamic=True)
    
    def warmup(self):
        # warmup
        self.fabric.print("Warming up the model...", file=sys.stderr)
        t = perf_counter()
        prompts = ["你好啊", "介绍一下北京", "介绍一下你自己"]
        for prompt in prompts:
            for token in self.generate(prompt=prompt):
                pass
        self.fabric.print(f"Time for warmup: {perf_counter() - t:.2f} seconds")
        self.fabric.print("\n")

    def prefill(self, **model_inputs) -> torch.Tensor:
        logits = self.model(**model_inputs)[0, -1]
        idx = self.sampler.sample(logits=logits)
        return idx

    def decode(self, **model_inputs) -> torch.Tensor:
        logits = self.model(**model_inputs)[0, -1]
        idx = self.sampler.sample(logits=logits)
        return idx

    @torch.inference_mode()
    def run(
        self,
        input_ids: torch.Tensor,
        stop_ids: List[torch.Tensor] | None = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Generator[torch.Tensor, None, None]:
        # 确保输入在设备上
        input_ids = self.fabric.to_device(input_ids)
        if input_pos is None:
            input_pos = self.fabric.to_device(torch.arange(len(input_ids)))
        if stop_ids is None:
            stop_ids = self.tokenizer.stop_ids
        stop_ids = [
            self.fabric.to_device(stop_id) for stop_id in self.tokenizer.stop_ids
        ]

        max_length = self.max_length if self.max_length else self.model.block_size

        # prefill
        input_ids = self.prefill(input_ids=input_ids.view(1, -1), input_pos=input_pos)
        yield input_ids

        # decode
        with self.fabric.init_tensor():
            input_pos = torch.tensor([input_pos[-1].item() + 1])
        max_stop_len = max([len(stop_id) for stop_id in stop_ids])
        yield_ids = []  # 用来存储的生成token ids, 直到长度达到stop ids中的最大长度, 然后生成
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            for i in range(1, max_length - input_pos.item() + 1):
                input_ids = input_ids.view(1, -1)
                next_token_id = self.decode(input_ids=input_ids, input_pos=input_pos)
                yield_ids.append(next_token_id)
                # 遍历每一个stop ids
                for ids in stop_ids:
                    if len(yield_ids) == len(ids):
                        if all(a == b for a, b in zip(yield_ids, ids)):
                            return
                if len(yield_ids) >= max_stop_len:
                    yield from yield_ids
                    yield_ids = []
                input_pos = input_pos.add_(1)
                input_ids = next_token_id

    def generate(self, prompt: str, sampler: Optional[Sampler] = None) -> Iterable[str]:
        messages = [Message(role="user", content=prompt)]
        yield from self.chat(messages=messages, sampler=sampler)

    def chat(
        self, messages: List[Message], sampler: Optional[Sampler] = None
    ) -> Iterable[str]:
        if sampler:
            self.sampler = sampler
        input_ids = self.tokenizer.encode_messages(messages)
        stream = self.run(input_ids=input_ids)
        for token in self.tokenizer.decode_stream(stream=stream):
            yield token
