from .base import LLMEngine
from ..utils import build_from_checkpoint
import torch
from typing import Generator, Optional, Tuple, List


class LLMEngineV3(LLMEngine):
    """v3版本的引擎
    特点:
    - 推测性解码
    """

    def load_model(self) -> None:
        self.model = build_from_checkpoint(checkpoint_dir=self.checkpoint_dir)
        self.fabric.to_device(self.model)
        with self.fabric.init_tensor():
            self.model.setup_kv_cache(batch_size=1, max_length=self.max_length)

        self.draft_model = build_from_checkpoint(checkpoint_dir=self.draft_checkpoint_dir)
        self.fabric.to_device(self.draft_model)
        with self.fabric.init_tensor():
            self.draft_model.setup_kv_cache(batch_size=1, max_length=self.max_length)

    def compile_model(self) -> None:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = (
            True  # Experimental feature to reduce compilation times, will be on by default in future
        )
        torch._inductor.config.triton.cudagraph_trees = False  # 目前用作server的时候有bug

        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        self.model = torch.compile(self.model, dynamic=True, fullgraph=True, mode="reduce-overhead")
        self.draft_model = torch.compile(self.draft_model, dynamic=True, fullgraph=True, mode="reduce-overhead")

    def setup_model(self) -> None:
        self.model = self.fabric.setup_module(self.model)
        self.draft_model = self.fabric.setup_module(self.draft_model)

    def run(
        self,
        input_ids: torch.Tensor,
        stop_ids: List[torch.Tensor],
        input_pos: Optional[torch.Tensor] = None,
        speculate_k: Optional[int] = 8,
    ) -> Generator[torch.Tensor, None, None]:
        # 确保输入在设备上
        input_ids = self.fabric.to_device(input_ids)
        if input_pos is None:
            input_pos = self.fabric.to_device(torch.arange(len(input_ids)))
        stop_ids = [self.fabric.to_device(stop_id) for stop_id in stop_ids]

        self.max_length = (
            self.max_length if self.max_length else min(self.model.block_size, self.draft_model.block_size)
        )

        # prefill
        input_ids = input_ids.view(1, -1)
        input_ids = self.prefill_target(input_ids=input_ids, input_pos=input_pos)
        yield input_ids
        self.prefill_draft(input_ids=input_ids, input_pos=input_pos)

        input_ids = input_ids.view(1, -1)
        with self.fabric.init_tensor():
            input_pos = torch.tensor([input_pos[-1].item() + 1])
        max_stop_len = max([len(stop_id) for stop_id in stop_ids])
        buffer = []
        while input_pos.item()[-1] < self.max_length:
            # decode
            draft_ids, draft_probs = self.speculative_decode_k(k=speculate_k, input_ids=input_ids, input_pos=input_pos)
            next_ids = self.verify(
                draft_ids=draft_ids,
                draft_probs=draft_probs,
                input_ids=input_ids,
                input_pos=input_pos,
            )
            for next_id in next_ids:
                if len(buffer) < max_stop_len:
                    buffer.append(next_id)
                    for ids in stop_ids:
                        if len(ids) == len(buffer) and all(a == b for a, b in zip(ids, buffer)):
                            return
                else:
                    yield from buffer
                    buffer = []
            input_ids = next_ids[-1]
            input_pos = input_pos.add_(len(next_ids))

    def prefill_target(self, **model_inputs) -> torch.Tensor:
        logits = self.model(**model_inputs)[0, -1]
        idx = self.sampler.sample(logits=logits)
        return idx

    def prefill_draft(self, **model_inputs) -> None:
        _ = self.draft_model(**model_inputs)

    def speculative_decode_k(self, k: int, **model_inputs) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        new_ids = []
        new_probs = []
        for i in range(k):
            ids, probs = self.speculate_next_token(**model_inputs)
            new_ids.append(ids)
            new_probs.append(probs)
        return torch.cat(new_ids), torch.cat(new_probs)

    def speculate_next_token(self, **model_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.draft_model(**model_inputs)[0, -1]
        probs = self.sampler.logits_to_probs(logits=logits)
        ids = self.sampler.probs_to_ids(probs=probs)
        return ids, probs

    def verify(
        self,
        cur_ids: torch.Tensor,
        cur_pos: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = torch.cat([cur_ids, draft_ids])
        with self.fabric.init_tensor():
            input_pos = torch.arange(cur_pos[-1].item(), cur_pos[-1].item() + 1 + len(draft_ids))

        target_logits = self.model(input_ids=input_ids.view(1, -1), input_pos=input_pos)[0]
        target_probs = self.sampler.logits_to_probs(logits=target_logits)

        q = target_probs.gather(dim=-1, index=draft_ids)
        p = draft_probs.gather(dim=-1, index=draft_ids)

        # 如果 q/p > 1, 则接受, 否则以 q/p 的概率接受
        accept_probs = torch.minimum(torch.ones(()), q / p)
        rejected_locations = (torch.rand_like(accept_probs) > accept_probs).nonzero()

        if rejected_locations.shape[0] == 0:
            last_token = self.sampler.multinomial_sample_one(probs=target_probs)
            _ = self.draft_model(input_ids=last_token, input_pos=input_pos)
            return torch.cat([draft_ids, last_token])

        else:
            pass
