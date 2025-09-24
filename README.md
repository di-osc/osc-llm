# OSC-LLM

轻量级大模型推理工具，专注于模型推理延迟。

## 安装

- 安装(最新版本pytorch)[https://pytorch.org/]
- 安装(flash-attn)[https://github.com/Dao-AILab/flash-attention]：建议下载官方构建好的whl包，避免编译问题
- 安装osc-llm
```bash
pip install osc-llm --upgrade
```

## 快速开始

### 下载模型

```bash
llm download Qwen/Qwen3-0.6B
```

### 基本使用

```python
from osc_llm import Qwen3ForCausalLM, Message

# 初始化模型
llm = Qwen3ForCausalLM("checkpoints/Qwen/Qwen3-0.6B")
llm.setup(device="cuda:0", gpu_memory_utilization=0.9)

# 对话
messages = [Message(role="user", content="介绍一下北京")]
messages = llm.chat(messages=messages, enable_thinking=True)
print(messages)
```

### 流式生成

```python
messages = [Message(role="user", content="介绍一下北京")]
for token in llm.chat(messages=messages, stream=True):
    print(token, end="", flush=True)
```

## 支持的模型

- Qwen3ForCausalLM (支持思考模式)

## CLI 工具

```bash
llm download <repo_id> [--endpoint hf-mirror|modelscope]
```