import json
from pathlib import Path
from typing import Any

from jinja2 import Template
from loguru import logger


class PromptTemplate:
    """Jinja2 prompt template"""

    def __init__(self, template: str) -> None:
        """初始化模板

        Args:
            template (str): 模板字符串
        """
        self.template: Template = Template(template)

    def render(self, **kwargs: Any) -> str:
        """渲染模板

        Args:
            **kwargs: 模板参数

        Returns:
            str: 渲染后的模板
        """
        return self.template.render(**kwargs)

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str) -> "PromptTemplate | None":
        """从LLM checkpoint目录加载Jinja2格式文件

        Args:
            checkpoint_dir (str): LLM checkpoint目录

        Returns:
            PromptTemplate: Jinja2格式文件
        """
        template_path = Path(checkpoint_dir) / "tokenizer_config.json"
        with open(template_path, "r") as f:
            template = json.load(f)
        chat_template = template.get("chat_template", None)
        if chat_template is None:
            logger.warning("No Jinja2 chat template found in the checkpoint directory")
            return None
        return cls(chat_template)
