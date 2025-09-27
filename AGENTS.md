## Project Description

This project is a lightweight large language model inference toolkit built on top of osc-transformers. It currently supports single-GPU inference. The goal is to provide a low-latency LLM inference service to power other AI systems such as speech generation, music generation, and real-time voice applications.


### Tech Stack

- [osc-transformers](https://github.com/di-osc/osc-transformers)
- [flash-attn](https://github.com/Dao-AILab/flash-attention)


### CausalLM

Provides end-to-end control for building models from HF checkpoints to osc-transformers. To add a new model, inherit from the base class and implement the `weight_map` and `osc_config` properties. The `weight_map` maps HF checkpoint weights to osc-transformers weights, and `osc_config` constructs the osc-transformers configuration.


### Development Rules

- **Python Code Style**: Use `ruff` for code formatting and linting. Run `ruff format` to format code and `ruff check` to check for issues.
- **Git Workflow**: Always run `ruff format` and `ruff check` before committing code changes to ensure code quality and consistency.

