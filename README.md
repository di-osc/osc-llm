<div align='center'>

# OSC-LLM
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/docs/overview/getting-started"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>

</div>

## 简介

osc-llm是一款轻量级别的模型推理框架, 专注于多模态推理的延迟和吞吐量。

## 特点

- ✅ 延迟低：torch.compile，cuda gragh
- ✅ 吞吐量高：PageAttention
- ✅ 支持多模态推理：llm，tts等
- ✅ 模型量化：WeightOnlyInt8，WeightOnlyInt4

> 文档地址:
- [notion](https://wangmengdi.notion.site/OSC-LLM-5a04563d88464530b3d32b31e27c557a)

## 安装

- 安装[最新版本pytorch](https://pytorch.org/get-started/locally/)
- 安装[flash-attention](https://github.com/Dao-AILab/flash-attention)
- 安装osc-llm: `pip install osc-llm`

## 快速开始

```python
from osc_llm import LLM

llm = LLM(model="checkpoints/Qwen/Qwen3-0.6B")
# 支持批量生成
outputs = llm.generate(prompts=["介绍一下你自己"])
# 支持流式生成
for token in llm.stream(prompt="介绍一下你自己"):
    print(token)
```

## 模型支持

LLM模型支持:
- **Qwen2ForCausalLM**: qwen1.5, qwen2等。
- **Qwen3ForCausalLM**: qwen3等。

TTS模型支持:
- **SparkTTS**: todo


### 致敬
本项目参考了大量的开源项目，特别是以下项目：

- [nanovllm](https://github.com/GeeeekExplorer/nano-vllm)
- [litgpt](https://github.com/Lightning-AI/litgpt)
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast)