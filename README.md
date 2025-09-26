# OSC-LLM

A lightweight LLM inference toolkit focused on minimizing inference latency.

[Chinese README](./readme-zh.md)

## Features

- **CUDA Graph**: Compilation optimizations that reduce inference latency
- **PagedAttention**: Efficient KV-cache management enabling long-sequence inference
- **Continuous batching**: Supports dynamic batch inference optimization

## Installation

- Install the [latest PyTorch](https://pytorch.org/)
- Install [flash-attn](https://github.com/Dao-AILab/flash-attention): recommended to use the official prebuilt wheel to avoid build issues
- Install osc-llm
```bash
pip install osc-llm --upgrade
```

## Quick Start


### Basic Usage

```python
from osc_llm import LLM, SamplingParams

# Initialize the model
llm = LLM("checkpoints/Qwen/Qwen3-0.6B", gpu_memory_utilization=0.5, device="cuda:0")

# Chat
messages = [
    {"role": "user", "content": "Hello! What's your name?"}
]
sampling_params = SamplingParams(temperature=0.5, top_p=0.95, top_k=40)
result = llm.chat(messages=messages, sampling_params=sampling_params, enable_thinking=True, stream=False)
print(result)

# Streaming generation
for token in llm.chat(messages=messages, sampling_params=sampling_params, enable_thinking=True, stream=True):
    print(token, end="", flush=True)
```

## Supported Models

- Qwen3ForCausalLM
- Qwen2ForCausalLM