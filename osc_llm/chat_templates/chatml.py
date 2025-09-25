from .base import ChatTemplate, Message
from ..registry import Registry
from typing import List


@Registry.chat_templates.register("Yi")
@Registry.chat_templates.register("ChatML")
@Registry.chat_templates.register("Qwen2ForCausalLM")
@Registry.chat_templates.register("Qwen/Qwen2.5-0.5B-Instruct")
class ChatMLChatTemplate(ChatTemplate):
    default_system: str = "You are a helpful assistant."
    stop_texts: List[str] = ["<|im_end|>"]
    generate_prompt: str = "<|im_start|>assistant\n"

    def apply_messages(
        self,
        messages: List[Message],
        add_generate_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        prompt = ""
        for message in messages:
            if message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
            elif message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        if add_generate_prompt:
            prompt += self.generate_prompt
        return prompt


@Registry.chat_templates.register("Qwen3ForCausalLM")
@Registry.chat_templates.register("Qwen/Qwen3-0.6B")
class Qwen3ChatTemplate(ChatMLChatTemplate):
    default_system: str = "You are a helpful assistant."
    stop_texts: List[str] = ["<|im_end|>"]
    generate_prompt: str = "<|im_start|>assistant\n"

    def apply_messages(
        self,
        messages: List[Message],
        add_generate_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        prompt = ""
        if messages and messages[0].role != "system":
            prompt += f"<|im_start|>system\n{self.default_system}<|im_end|>\n"
        for message in messages:
            if message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
            elif message.role == "system":
                prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        if add_generate_prompt:
            prompt += self.generate_prompt
        if not enable_thinking:
            prompt += "<think>\n\n</think>\n\n"
        return prompt
