# OSC-LLM

osc-llm旨在成为一个简单易用的大模型训练、评估、部署工具，支持目前主流的大模型。

### 安装

- 安装pytorch: 
- 安装osc-llm: `pip install osc-llm`

### 项目特点

- 简单易用的命令行系统
- 基于python entrypoint的配置系统,可以轻松拓展本项目的所有模块

### 支持功能

- [ ] 全参数微调sft
- [ ] 高效微调sft
- [ ] 大模型评估
- [ ] 大模型部署

### Huggingface模型支持

- **LlamaForCausalLM**: llama2, llama2, chinese-alpaca2等。
- **Qwen2ForCausalLM**: qwen1.5系列等。


### 致敬
本项目参考了大量的开源项目，特别是以下项目：

- [litgpt](https://github.com/Lightning-AI/litgpt)
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast)