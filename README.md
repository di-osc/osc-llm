<div align='center'>

# OSC-LLM
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/docs/overview/getting-started"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>

</div>

## 简介

osc-llm是一款轻量级别的模型推理框架, 专注于易用性和多任务的推理。

## 特点

- 使用torch.compile减少最多４倍以上的推理时间。
- 使用int8,int4量化减少显存占用。
- 使用Speculative decoding减少推理时间。

> 文档地址:
- [notion](https://wangmengdi.notion.site/OSC-LLM-5a04563d88464530b3d32b31e27c557a)

## 安装

- 安装[最新版本pytorch](https://pytorch.org/get-started/locally/)
- 安装osc-llm: `pip install osc-llm`

## 快速开始
```bash
# 下面以llama3为例演示如何转换为osc-llm格式,并进行聊天。
# 假设你已经下载好huggingface的llama3模型在checkpoints/meta-llama目录下
# 1. 转换
llm convert --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
# 2. 量化
llm quantize int8 --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct --save_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct-int8
# 3. 聊天(使用编译功能加速推理速度,需要等待几分钟编译时间)
llm chat --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct-int8 --compile true
# 4. 部署
llm serve --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct-int8
```

## 模型支持

以下huggingface中的模型结构(查看config.json)已经支持转换为osc-llm格式:
- **LlamaForCausalLM**: llama2, llama3, chinese-alpaca2等。
- **Qwen2ForCausalLM**: qwen1.5系列。
- **Qwen2MoeForCausalLM**: qwen2-moe系列(目前无法完成编译,推理速度很慢)。


### 致敬
本项目参考了大量的开源项目，特别是以下项目：

- [litgpt](https://github.com/Lightning-AI/litgpt)
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast)