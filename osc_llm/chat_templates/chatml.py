from .base import ChatTemplate, Message
from ..config import registry
from typing import List



@registry.chat_templates.register("Yi")
@registry.chat_templates.register("Qwen")
@registry.chat_templates.register("ChatML")
class ChatMLChatTemplate(ChatTemplate):
    
    default_system: str = "You are a helpful assistant."
    stop_texts: List[str] = ["<|im_end|>"]
    
    @classmethod
    def apply_messages(cls, messages: List[Message]) -> str:
        prompt = ""
        for message in messages:
            if message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
            elif message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        return prompt