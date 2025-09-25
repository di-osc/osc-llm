from __future__ import annotations

from typing import List, Literal, Optional, Dict, Tuple

from pydantic import BaseModel, model_validator
from pathlib import Path
from confection import Config
from loguru import logger

from ..registry import Registry


class Property(BaseModel):
    type: str
    description: Optional[str] = None


class Parameters(BaseModel):
    type: str
    properties: Dict[str, Property]
    required: List[str]


class Tool(BaseModel):
    name: str
    description: str
    parameters: Parameters


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "observation"]
    thinking_content: Optional[str] = None
    content: str
    metadata: str = ""
    tools: List[Tool] = []


class ChatTemplate(BaseModel):
    default_system: Optional[str] = None
    stop_texts: List[str] = []
    generate_prompt: str = None
    messages: List[Message] = []

    @model_validator(mode="after")
    def add_default_system(self):
        if self.default_system:
            self.messages.append(Message(role="system", content=self.default_system))
        return self

    def apply(self, enable_thinking: bool = True) -> str:
        if self.messages[-1].role != "user":
            logger.warning("Last message is not user, please add user message")
        if self.messages:
            return self.apply_messages(
                self.messages, add_generate_prompt=True, enable_thinking=enable_thinking
            )
        return ""

    def apply_messages(
        self,
        messages: List[Message],
        add_generate_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        raise NotImplementedError

    def add_user_message(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str):
        thinking_content, content = self.split_thinking_content(content)
        self.messages.append(
            Message(
                role="assistant", content=content, thinking_content=thinking_content
            )
        )

    def reset_system(self, content: str):
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = content
        else:
            self.messages.insert(0, Message(role="system", content=content))

    def split_thinking_content(self, content: str) -> Tuple[str, str]:
        import re

        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            thinking_content = match.group(1)
            thinking_content = f"<think>{thinking_content}</think>"
            content = content.replace(thinking_content, "")
            return thinking_content.strip().strip("\n"), content.strip().strip("\n")
        return "", content

    def reset(self):
        self.messages = []
        self.messages.append(Message(role="system", content=self.default_system))

    @classmethod
    def get_config(cls) -> Config:
        for k, v in Registry.chat_templates.get_all().items():
            if cls == v:
                name = k
        config_str = f"""
        [chat_template]
        @chat_templates = {name}"""
        config = Config().from_str(config_str)
        return config

    @classmethod
    def from_name(cls, name: str) -> ChatTemplate:
        template_cls = None
        for k, v in Registry.chat_templates.get_all().items():
            if k in name:
                template_cls = v
        return template_cls

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str | Path) -> ChatTemplate:
        checkpoint_dir = Path(checkpoint_dir)
        # model_name : Qwen/Qwen3-0.6B
        model_name = checkpoint_dir.parent.name + "/" + checkpoint_dir.name
        return Registry.chat_templates.get(model_name)

    @classmethod
    def from_hf_architecture(cls, architecture: str) -> ChatTemplate:
        # architecture: Qwen3ForCausalLM
        return Registry.chat_templates.get(architecture)
