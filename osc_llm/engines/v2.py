from .base import LLMEngine
from ..utils import build_model
from ..architectures import TransformerDecoder
from ..config import Config, registry
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import List, Generator, Optional
from pathlib import Path


@registry.engines.register("v2.TransformerDecoder")
@registry.engines.register("v2")
class LLMEngineV2(LLMEngine):
    """v2版本语言模型引擎
    特点:
    1. 基于torch.compile进行模型编译,大大的提升了模型推理速度。
    2. 为了解决每次推理都需要重新编译的问题,在prefill和decode两个阶段分别使用两份相同的模型,并且共享kv_cache和mask_cache。
    3. prefill模型用动态形状编译,decode模型用静态编译。

    注意:
    1. torch.compile机制会在首次运行时候进行真正的编译,因此首次运行会比较慢。
    2. 建议先运行至少2次run方法,以便编译完成。
    """

    def load_model(self) -> None:
        config_path = Path(self.checkpoint_dir) / "config.cfg"
        states_path = Path(self.checkpoint_dir) / "osc_model.pth"

        config = Config().from_disk(config_path)
        assert (
            config["model"]["@architectures"] == "TransformerDecoder"
        ), "Only TransformerDecoder Architecture is supported"

        with self.fabric.init_module(empty_init=True):
            self.prefill_model = build_model(config=config_path, empty_init=False).eval()
            self.decode_model = build_model(config=config_path, empty_init=False).eval()

        self.fabric.load_raw(states_path, self.prefill_model)
        self.fabric.load_raw(states_path, self.decode_model)

        with self.fabric.init_tensor():
            self.prefill_model.setup_kv_cache(batch_size=1, max_length=self.max_length, dtype=torch.bfloat16)

        self.decode_model.kv_caches = self.prefill_model.kv_caches
        self.decode_model.mask_cache = self.prefill_model.mask_cache

    def compile_model(self) -> None:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = (
            True  # Experimental feature to reduce compilation times, will be on by default in future
        )
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.suppress_errors = True
        self.decode_model: TransformerDecoder = torch.compile(self.decode_model, mode="reduce-overhead", fullgraph=True)
        self.prefill_model: TransformerDecoder = torch.compile(self.prefill_model, dynamic=True)

    def setup_model(self) -> None:
        self.prefill_model = self.fabric.setup_module(self.prefill_model)
        self.decode_model = self.fabric.setup_module(self.decode_model)

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

        # prefill
        max_length = self.max_length if self.max_length else self.prefill_model.block_size
        input_ids = self.prefill(input_ids=input_ids.view(1, -1), input_pos=input_pos)
        yield input_ids

        # decode
        with self.fabric.init_tensor():
            input_pos = torch.tensor([input_pos[-1].item() + 1])
        max_stop_len = max([len(stop_id) for stop_id in stop_ids])
        yield_ids = []
        for i in range(1, max_length - input_pos.item() + 1):
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                input_ids = input_ids.view(1, -1)
                next_token_id = self.decode(input_ids=input_ids, input_pos=input_pos)
                yield_ids.append(next_token_id)
                # 遍历每一个stop ids
                for ids in stop_ids:
                    if len(yield_ids) >= len(ids):
                        if all(a == b for a, b in zip(yield_ids, ids)):
                            return
                if len(yield_ids) >= max_stop_len:
                    yield from yield_ids
                    yield_ids = []
            input_pos = input_pos.add_(1)
            input_ids = next_token_id

    @torch.inference_mode()
    def prefill(self, **model_inputs) -> torch.Tensor:
        logits = self.prefill_model(**model_inputs)[0, -1]
        idx = self.sampler.sample(logits=logits)
        return idx

    @torch.inference_mode()
    def decode(self, **model_inputs) -> torch.Tensor:
        logits = self.decode_model(**model_inputs)[0, -1]
        idx = self.sampler.sample(logits=logits)
        return idx
