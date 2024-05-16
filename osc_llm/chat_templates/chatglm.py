from .base import ChatTemplate, Message
from typing import List
import json
from ..config import registry


@registry.chat_templates.register("chatglm3")
class ChatGLM3ChatTemplate(ChatTemplate):
    default_system: str = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
    stop_texts: List[str] = ["<|observation|>", "<|user|>"]

    @classmethod
    def apply_messages(cls, messages: List[Message], add_generate_prompt: bool = True) -> str:
        prompt: str = ""
        for message in messages:
            if message.role == "user":
                prompt += f"<|user|>{message.metadata}\n{message.content}"
            elif message.role == "assistant":
                prompt += f"<|assistant|>{message.metadata}\n{message.content}"
            elif message.role == "system":
                content = message.content
                if len(message.tools) > 0:
                    content += "\n"
                    content += json.dumps(message.model_dump()["tools"])
                prompt += f"<|system|>{message.metadata}\n{content}"
        if add_generate_prompt:
            prompt += "<|assistant|>"
        return prompt
