from typing import List
from .base import ChatTemplate, Message
from ..config import registry


@registry.chat_templates.register("Llama3-8B-Chinese-Chat")
@registry.chat_templates.register("llama-3-chinese-8b-instruct")
@registry.chat_templates.register("Llama-3")
class Llama3ChatTemplate(ChatTemplate):
    stop_texts: List[str] = ["<|end_of_text|>", "<|eot_id|>"]
    generate_prompt: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    @classmethod
    def apply_messages(cls, messages: List[Message], add_generate_prompt: bool = False) -> str:
        assert messages[-1].role == "user", "Last message must be user"
        prompt = "<|begin_of_text|>"
        for message in messages:
            prompt += _apply_message_llama3(message)
        if add_generate_prompt:
            prompt += cls.generate_prompt
        return prompt


def _apply_message_llama3(message: Message) -> str:
    template = "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    return template.format(role=message.role, content=message.content)


@registry.chat_templates.register("Llama-2")
class Llama2ChatTemplate(ChatTemplate):
    default_system: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as"
        " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
        " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
        " instead of answering something not correct. If you don't know the answer to a question, please don't"
        " share false information."
    )
    stop_texts: List[str] = []
    generate_prompt: str = ""

    @classmethod
    def apply_messages(cls, messages: List[Message], add_generate_prompt: bool = True) -> str:
        if messages[0].role == "system":
            assert len(messages) >= 2, "must have a user input"
            assert messages[1].role == "user", "must have a user input"
            prompt = f"[INST] <<SYS>>\n{messages[0].content}\n<</SYS>>\n\n{messages[1].content} [/INST]"
            for message in messages[2:]:
                prompt += _apply_message_llama2(message)
        else:
            prompt = f"[INST] <<SYS>>\n{cls.default_system}\n<</SYS>>\n\n{messages[0].content} [/INST]"
            for message in messages[1:]:
                prompt += _apply_message_llama2(message)
        return prompt


def _apply_message_llama2(message: Message) -> str:
    if message.role == "user":
        return f"[INST] {message.content} [/INST]"
    if message.role == "assistant":
        return f" {message.content} "


@registry.chat_templates.register("chinese-alpaca-2")
class ChineseAlpaca2ChatTemplate(Llama2ChatTemplate):
    default_system: str = "You are a helpful assistant, 你是一个乐于助人的助手."
