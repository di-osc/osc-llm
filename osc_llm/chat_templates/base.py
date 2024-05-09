from typing import List, Literal, Optional
from pydantic import BaseModel
from pathlib import Path
from ..config import registry, Config



class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    

class ChatTemplate():
    
    default_system: Optional[str] = ''
    stop_texts: List[str] = []
    generate_prompt: str = ''
    
    @classmethod
    def apply_messages(cls, messages: List[Message], add_generate_prompt: bool = True) -> str:
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
            if isinstance(cls, v):
                name = k
        config_str = f"""
        [chat_template]
        @chat_templates = {name}"""
        config = Config().from_str(config_str)
        return config
    
    @classmethod
    def from_name(cls, name: str) -> "ChatTemplate":
        template_cls = None
        for k, v in registry.chat_templates.get_all().items():
            if k in name:
                template_cls = v
        return template_cls
    
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "ChatTemplate":
        checkpoint_dir = Path(checkpoint_dir)
        return cls.from_name(checkpoint_dir.stem)