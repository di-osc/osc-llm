<div align='center'>

# OSC-LLM
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/docs/overview/getting-started"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>

</div>

## 简介

osc-llm是一款轻量级别的模型推理框架, 专注于延迟和易用性。

## 特点

- ✅ 模型编译：torch.compile
- ✅ 模型量化：WeightOnlyInt8，WeightOnlyInt4
- ✅ 推测性解码：（Speculative decoding）
- ✅ 极少的依赖：核心仅pytorch
- ❎ 更高效的kvcache管理：（PageAttention、TokenAttention）

> 文档地址:
- [notion](https://wangmengdi.notion.site/OSC-LLM-5a04563d88464530b3d32b31e27c557a)

## 安装

- 安装[最新版本pytorch](https://pytorch.org/get-started/locally/)
- 安装osc-llm: `pip install osc-llm`

## 快速开始

命令行
```bash
# 聊天(使用编译功能加速推理速度,需要等待几分钟编译时间)
llm chat --checkpoint_dir checkpoints/Qwen/Qwen3-0.6B --compile true
# 部署openai api
llm serve --checkpoint_dir checkpoints/Qwen/Qwen3-0.6B --compile true
```
使用LLM
```python
from osc_llm import LLM

llm = LLM(checkpoint_dir="checkpoints/Qwen/Qwen3-0.6B", compile=True)
for token in llm.generate(prompt="介绍一下你自己"):
    print(token)
```

## 模型支持

以下huggingface中的模型(查看config.json)已经得到支持:
- **LlamaForCausalLM**: llama2, llama3, chinese-alpaca2等。
- **Qwen2ForCausalLM**: qwen1.5, qwen2等。
- **Qwen2MoeForCausalLM**: qwen2-moe系列(目前无法完成编译,推理速度很慢)。
- **Qwen3ForCausalLM**: qwen3等。


### 致敬
本项目参考了大量的开源项目，特别是以下项目：

- [litgpt](https://github.com/Lightning-AI/litgpt)
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast)