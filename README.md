# OSC-LLM

轻量级大模型推理工具，专注于模型推理延迟。

## 特性

- **CUDA Graph**: 编译优化，减少推理延迟
- **PagedAttention**: 高效的KV缓存管理，支持长序列推理
- **连续批处理**: 支持动态批量推理优化

## 安装

- 安装[最新版本pytorch](https://pytorch.org/)
- 安装[flash-attn](https://github.com/Dao-AILab/flash-attention): 建议下载官方构建好的whl包，避免编译问题
- 安装osc-llm
```bash
pip install osc-llm --upgrade
```

## 快速开始


### 基本使用

```python
from osc_llm import LLM, SamplingParams

# 初始化模型
llm = LLM("checkpoints/Qwen/Qwen3-0.6B", gpu_memory_utilization=0.5, device="cuda:0")

# 对话
messages = [
    {"role": "user", "content": "你好啊，你叫什么?"}
]
sampling_params = SamplingParams(temperature=0.5, top_p=0.95, top_k=40)
result = llm.chat(messages=messages, sampling_params=sampling_params, enable_thinking=True, stream=False)
print(result)

# 流式生成
for token in llm.chat(messages=messages, sampling_params=sampling_params, enable_thinking=True, stream=True):
    print(token, end="", flush=True)
```

## 支持的模型

- Qwen3ForCausalLM
- Qwen2ForCausalLM