from .base import LLMEngine
from ..utils import get_hf_model_helper
from ..architectures import TransformerDecoder
from ..config import registry
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import List, Generator, Optional


@registry.engines.register("v1.TransformerDecoder")
@registry.engines.register("v1")
class LLMEngineV1(LLMEngine):
    """v1版本语言模型引擎
    特点:
    1. 基于torch.compile进行``动态形状``模型编译,大大的提升了模型推理速度。

    注意:
    1. torch.compile机制会在首次运行时候进行真正的编译,因此首次运行会比较慢。
    2. 建议先运行至少2次run方法,以便编译完成。
    """

    def load_model(self) -> None:
        model_helper = get_hf_model_helper(self.checkpoint_dir)
        self.model: TransformerDecoder = model_helper.load_model()
        self.model = self.fabric.to_device(self.model)

        with self.fabric.init_tensor():
            self.model.setup_kv_cache(
                batch_size=1, max_length=self.max_length, dtype=torch.bfloat16
            )

    def compile_model(self) -> None:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
        torch._inductor.config.triton.cudagraph_trees = (
            False  # 目前用作server的时候有bug
        )
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        self.decode = torch.compile(self.decode, fullgraph=True, mode="reduce-overhead")
        self.prefill = torch.compile(self.prefill, dynamic=True, fullgraph=True)

    def setup_model(self) -> None:
        self.model = self.fabric.setup_module(self.model)

    @torch.inference_mode()
    def run(
        self,
        input_ids: torch.Tensor,
        stop_ids: List[torch.Tensor],
        input_pos: Optional[torch.Tensor] = None,
    ) -> Generator[torch.Tensor, None, None]:
        # 确保输入在设备上
        input_ids = self.fabric.to_device(input_ids)
        if input_pos is None:
            input_pos = self.fabric.to_device(torch.arange(len(input_ids)))
        stop_ids = [self.fabric.to_device(stop_id) for stop_id in stop_ids]

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

    def prefill(self, **model_inputs) -> torch.Tensor:
        logits = self.model(**model_inputs)[0, -1]
        idx = self.sampler.sample(logits=logits)
        return idx

    def decode(self, **model_inputs) -> torch.Tensor:
        logits = self.model(**model_inputs)[0, -1]
        idx = self.sampler.sample(logits=logits)
        return idx
