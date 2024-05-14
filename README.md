# OSC-LLM

osc-llm旨在成为一个简单易用的大模型训练、评估、部署工具，支持目前主流的大模型。

### 项目特点

- 简单易用的命令行系统
- 基于python entrypoint的配置系统,可以轻松拓展本项目的所有模块

### 安装

- 安装pytorch: 
- 安装osc-llm: `pip install osc-llm`

### 快速开始
```bash
# 下面以llama3为例演示如何转换为osc-llm格式,并进行聊天。
# 假设你已经下载好huggingface的llama3模型在checkpoints/meta-llama目录下
# 1. 转换模型
llm convert --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
# 2. 量化模型
llm quantize int8 --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct --save_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct-int8
# 3. 聊天(使用编译功能加速推理速度,需要等待几分钟编译时间)
llm chat --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct-int8 --compile true
# 4. 部署简易版本openai服务
llm serve --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B-Instruct-int8
```

### 支持功能

- [ ] 全参数微调sft
- [ ] 高效微调sft
- [ ] 大模型评估
- [ ] 大模型部署

### Huggingface模型支持

- **LlamaForCausalLM**: llama2, llama3, chinese-alpaca2等。
- **Qwen2ForCausalLM**: qwen1.5系列。
- **Qwen2MoeForCausalLM**: qwen2-moe系列(目前无法完成编译,推理速度很慢)。


### 致敬
本项目参考了大量的开源项目，特别是以下项目：

- [litgpt](https://github.com/Lightning-AI/litgpt)
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast)