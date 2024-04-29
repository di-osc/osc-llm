from .architectures import TransformerDecoder
from .tokenizer import Tokenizer
from .utils import build_model
from .chat_templates import Message
from .samplers import Sampler, TopK
from .config import registry
from .layers import StaticKVCache
from lightning import Fabric
import torch
import time
import sys
from pathlib import Path
from typing import Optional, List



torch.set_float32_matmul_precision('high')


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")

def prefill(model, input_ids: torch.Tensor, input_pos: torch.Tensor, sampler: Sampler) -> torch.Tensor:
    logits = model(input_ids.view(1, -1), input_pos)[0, -1]
    idx = sampler.sample(logits=logits)
    return idx

def decode_one_token(model, input_ids, input_pos, sampler: Sampler):
    logits = model(input_ids, input_pos)[0, -1]
    idx = sampler.sample(logits=logits)
    return idx
        

@torch.inference_mode()
def generate(
    input_ids: torch.Tensor,
    input_pos: torch.Tensor,
    prefill_model: TransformerDecoder,
    decode_model: TransformerDecoder,
    max_length: int,
    stop_ids: List[torch.Tensor],
    temperature: float = 1.0,
    top_k: int = 10,
):
    sampler = TopK(k=top_k, temperature=temperature)
    input_ids = prefill(prefill_model, input_ids, input_pos, sampler=sampler)
    yield input_ids
    input_pos = torch.tensor([input_pos[-1].item() + 1], device=input_ids.device)
    max_stop_len = max([len(stop_id) for stop_id in stop_ids])
    yield_ids = []
    for i in range(1, max_length - input_pos.item() + 1):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            input_ids = input_ids.view(1, -1)
            next_token_id = decode_one_token(model=decode_model, input_ids=input_ids, input_pos=input_pos, sampler=sampler)
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


def load_model(fabric: Fabric, checkpoint_dir: str):
    config_path = Path(checkpoint_dir) / 'config.cfg'
    states_path = Path(checkpoint_dir) / 'osc_model.pth'
        
    time0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        prefill_model = build_model(config=config_path, empty_init=False)
        decode_model = build_model(config=config_path, empty_init=False)
    time1 = time.perf_counter()
    fabric.print(f"build model in {time1 - time0:.02f} seconds", file=sys.stderr)
    
    time2 = time.perf_counter()
    fabric.load_raw(states_path, prefill_model)
    fabric.load_raw(states_path, decode_model)
    time3 = time.perf_counter()
    fabric.print(f"load state dict in {time3 - time2:.02f} seconds", file=sys.stderr)
    
    return prefill_model.eval(), decode_model.eval()
    

@torch.inference_mode()
def main(
    checkpoint_dir: str,
    device: int = 0,
    temperature: float = 1.0,
    top_k: int = 200,
    max_length: Optional[int] = None,
    compile: bool = False,
    compile_prefill: bool = False,
    multi_turn: bool = False
):
    """chat with llm

    Args:
        checkpoint_dir (str): the directory of the model checkpoint
        device (int, optional): which gpu to use. Defaults to 0.
        temperature (float, optional): the temperature of the sampling. Defaults to 1.0.
        top_k (int, optional): the top k sampling. Defaults to 200.
        max_length (Optional[int], optional): the max length of the generation. Defaults to None.
        compile (bool, optional): whether to use torch.compile. Defaults to False.
        compile_prefill (bool, optional): whether to compile prefill model. Defaults to False.
        multi_turn (bool, optional): whether to use multi-turn chat. Defaults to False.
    """
    checkpoint_dir: Path = Path(checkpoint_dir)
    fabric = Fabric(devices=[device], accelerator='cuda', precision='bf16-true')
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    
    prefill_model, decode_model = load_model(fabric, checkpoint_dir)
    with fabric.init_tensor():
        prefill_model.setup_kv_cache(batch_size=1, kv_cache=StaticKVCache(), max_length=max_length, dtype=torch.bfloat16)
    decode_model.kv_caches = prefill_model.kv_caches
    decode_model.mask_cache = prefill_model.mask_cache
    
    if not max_length:
        max_length = prefill_model.block_size
    fabric.print(f"model max length: {max_length}")
    
    if compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.suppress_errors = True
        fabric.print("Compiling model")
        decode_model: TransformerDecoder = torch.compile(decode_model, mode="reduce-overhead", fullgraph=True)
        if compile_prefill:
            prefill_model: TransformerDecoder = torch.compile(prefill_model, dynamic=True)
            
    decode_model = fabric.setup_module(decode_model)
    prefill_model = fabric.setup_module(prefill_model)
    
    for k, v in registry.chat_templates.get_all().items():
        if k in checkpoint_dir.stem:
            fabric.print(f"using {k} chat template")
            template = v
            
    with fabric.init_tensor():
        stop_token_ids = [tokenizer.encode(text) for text in template.stop_texts]
        stop_token_ids.append(torch.tensor([tokenizer.eos_id]))
    
    if compile:
        fabric.print("Warmup model")
        t = time.perf_counter()
        prompt = template.apply_user("你好")
        with fabric.init_tensor():
            input_ids = tokenizer.encode(prompt)
            input_pos = torch.arange(0, len(input_ids))
        y =  generate(input_ids=input_ids, 
                      input_pos=input_pos,
                      prefill_model=prefill_model, 
                      decode_model=decode_model, 
                      max_length=512, 
                      stop_ids=stop_token_ids)
        _ = tokenizer.decode_stream(stream=y)
        fabric.print(f"Time for warmup: {time.perf_counter() - t:.2f} seconds")
        fabric.print("\n")

    messages = []
    pre_ids_len = 0 # 多轮对话过程中,对之前的对话历史做一个缓存,这样避免在prefill阶段重新kv cache
    while True:
        content = input("User (empty to exit): ")
        if content == "":
            break
        
        messages.append(Message(role='user', content=content))
        prompt = template.apply_messages(messages)
        
        with fabric.init_tensor():
            input_ids = tokenizer.encode(prompt)
            input_pos = torch.arange(pre_ids_len, len(input_ids))
            input_ids = input_ids[pre_ids_len:]
        
        y = generate(input_ids=input_ids, 
                     input_pos=input_pos, 
                     prefill_model=prefill_model, 
                     decode_model=decode_model, 
                     max_length=max_length, 
                     stop_ids=stop_token_ids,
                     temperature=temperature, 
                     top_k=top_k)
        
        fabric.print("Assistant: ")
        time0 = time.perf_counter()
        generated_text = tokenizer.decode_stream(stream=y, print_stream=True)
        device_sync(device=f"cuda:{device}")
        time1 = time.perf_counter()
        t = time1 - time0
        num_new_tokens = len(tokenizer.encode(generated_text).tolist())
        
        if multi_turn:
            messages.append(Message(role='assistant', content=generated_text))
            pre_ids_len += len(input_ids)
        else:
            messages = []
            pre_ids_len = 0
        
        fabric.print("\n")
        fabric.print(f"Generated {num_new_tokens} tokens in {t:.02f} seconds, {(num_new_tokens / t):.2f} tokens/second", file=sys.stderr)
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
        fabric.print("\n")