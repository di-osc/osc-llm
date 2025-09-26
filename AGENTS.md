## 项目描述

该项目是一个基于osc-transformers构建的大模型推理工具，轻量级别，暂仅支持单卡推理，目标是构建低延迟的大模型推理服务，以支持其它基于LLM的AI推理系统，例如语音生成，音乐生成，以及实时语音等。


### 技术栈

- [osc-transformers](https://github.com/di-osc/osc-transformers)
- [flash-attn](https://github.com/Dao-AILab/flash-attention)


### CausalLM

实现从HF checkpoint到osc-transformers构建全流程控制，添加新模型继承该基类即可。一般情况下仅需要实现`weight_map`和`osc_config`属性，`weight_map`用于映射HF checkpoint的权重到osc-transformers的权重，`osc_config`用于构建osc-transformers的配置。




