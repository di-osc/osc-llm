from .tokenizer import Tokenizer
from .chat_templates import Message
from .samplers import TopK
from .engines import LLMEngineV1, LLMEngine, LLMEngineV2
import torch
import time
from pathlib import Path
from typing import Optional, Literal


torch.set_float32_matmul_precision("high")


@torch.inference_mode()
def main(
    checkpoint_dir: str,
    device: int = 0,
    temperature: float = 1.0,
    top_k: int = 200,
    max_length: Optional[int] = None,
    compile: bool = False,
    multi_turn: bool = False,
    engine: Literal["v1", "v2"] = "v1",
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
        engine (Literal["v1", "v2"], optional): which engine to use. Defaults to "v1".
    """
    checkpoint_dir: Path = Path(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    sampler = TopK(k=top_k, temperature=temperature)
    if engine == "v1":
        engine: LLMEngine = LLMEngineV1(
            checkpoint_dir=checkpoint_dir,
            sampler=sampler,
            max_length=max_length,
            devices=[device],
            compile=compile,
        )
    elif engine == "v2":
        engine: LLMEngine = LLMEngineV2(
            checkpoint_dir=checkpoint_dir,
            sampler=sampler,
            max_length=max_length,
            devices=[device],
            compile=compile,
        )
    engine.setup()

    if not hasattr(engine, "decode_model"):
        model_size = engine.model.model_size(include_embeddings=False)
    else:
        model_size = engine.decode_model.model_size(include_embeddings=False) + engine.prefill_model.model_size(
            include_embeddings=False
        )

    if compile:
        t = time.perf_counter()
        input_ids = tokenizer.encode_messages([Message(role="user", content="你好")])
        stream = engine.run(input_ids=input_ids, stop_ids=tokenizer.stop_ids)
        token_stream = tokenizer.decode_stream(stream=stream)
        for token in token_stream:
            pass
        engine.fabric.print(f"Time for warmup: {time.perf_counter() - t:.2f} seconds")
        engine.fabric.print("\n")

    messages = []
    pre_ids_len = 0  # 多轮对话过程中,对之前的对话历史做一个缓存,这样避免在prefill阶段重新kv cache
    while True:
        content = input("User (empty to exit): ")
        if content == "":
            break

        messages.append(Message(role="user", content=content))
        input_ids = tokenizer.encode_messages(messages)
        with engine.fabric.init_tensor():
            input_pos = torch.arange(pre_ids_len, len(input_ids))
        input_ids = input_ids[pre_ids_len:]

        stream = engine.run(input_ids=input_ids, stop_ids=tokenizer.stop_ids, input_pos=input_pos)
        generated_text = ""
        engine.fabric.print("Assistant: ")
        time0 = time.perf_counter()
        token_stream = tokenizer.decode_stream(stream=stream)
        for token in token_stream:
            print(token, end="", flush=True)
            generated_text += token
        time1 = time.perf_counter()
        t = time1 - time0
        num_new_tokens = len(tokenizer.encode(generated_text).tolist())
        tokens_sec = num_new_tokens / t

        if multi_turn:
            messages.append(Message(role="assistant", content=generated_text))
            pre_ids_len += len(input_ids)
        else:
            messages = []
            pre_ids_len = 0

        engine.fabric.print("\n")
        engine.fabric.print(
            f"Generated {num_new_tokens} tokens in {t:.02f} seconds, {(num_new_tokens / t):.2f} tokens/second"
        )
        engine.fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        engine.fabric.print(f"Bandwidth Achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        engine.fabric.print("\n")
