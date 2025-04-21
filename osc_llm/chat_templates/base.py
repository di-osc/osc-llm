from __future__ import annotations

from typing import List, Literal, Optional, Dict
from pydantic import BaseModel
from ..config import registry, Config


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
    content: str
    metadata: str = ""
    tools: List[Tool] = []


class ChatTemplate:
    default_system: Optional[str] = ""
    stop_texts: List[str] = []
    generate_prompt: str = ""

    @classmethod
    def apply_messages(
        cls, messages: List[Message], add_generate_prompt: bool = True
    ) -> str:
        raise NotImplementedError

    @classmethod
    def apply_user(cls, user: str, add_generate_prompt: bool = True) -> str:
        messages = []
        if cls.default_system:
            messages.append(Message(role="system", content=cls.default_system))
        messages.append(Message(role="user", content=user))
        return cls.apply_messages(messages, add_generate_prompt=add_generate_prompt)

    @classmethod
    def get_config(cls) -> Config:
        for k, v in registry.chat_templates.get_all().items():
            if cls == v:
                name = k
        config_str = f"""
        [chat_template]
        @chat_templates = {name}"""
        config = Config().from_str(config_str)
        return config

    @classmethod
    def from_name(cls, name: str) -> ChatTemplate | None:
        template_cls = None
        for k, v in registry.chat_templates.get_all().items():
            if k in name:
                template_cls = v
        return template_cls
