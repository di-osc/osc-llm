import json
from pathlib import Path
from typing import Any

from jinja2 import Template
from loguru import logger


class PromptTemplate:
    """Jinja2 prompt template"""

    def __init__(self, template: str) -> None:
        """Initialize the template.

        Args:
            template (str): Raw Jinja2 template string.
        """
        self.template: Template = Template(template)

    def render(self, **kwargs: Any) -> str:
        """Render the template with keyword arguments.

        Args:
            **kwargs: Variables to render into the template.

        Returns:
            str: The rendered string.
        """
        return self.template.render(**kwargs)

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str) -> "PromptTemplate | None":
        """Load a Jinja2 chat template from an LLM checkpoint directory.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.

        Returns:
            PromptTemplate | None: The prompt template if available, else None.
        """
        template_path = Path(checkpoint_dir) / "tokenizer_config.json"
        with open(template_path, "r") as f:
            template = json.load(f)
        chat_template = template.get("chat_template", None)
        if chat_template is None:
            logger.warning("No Jinja2 chat template found in the checkpoint directory")
            return None
        return cls(chat_template)
