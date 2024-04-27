from typing import List, Literal, Optional
from pydantic import BaseModel
from ..config import registry, Config
from ..tokenizer import Tokenizer



class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    

class ChatTemplate():
    
    default_system: Optional[str] = ''
    stop_texts: List[str] = []
    
    @classmethod
    def apply_messages(cls, messages: List[Message]) -> str:
        raise NotImplementedError
    
    @classmethod
    def apply_user(cls, user: str) -> str:
        messages = []
        if cls.default_system:
            messages.append(Message(role="system", content=cls.default_system))
        messages.append(Message(role="user", content=user))
        return cls.apply_messages(messages)
    
    @classmethod
    def get_stop_tokens(cls, tokenizer: Tokenizer):
        raise NotImplementedError

    def get_config(self) -> Config:
        for k, v in registry.chat_templates.get_all().items():
            if isinstance(self, v):
                name = k
        config_str = f"""
        [chat_template]
        @chat_templates = {name}"""
        config = Config().from_str(config_str)
        return config