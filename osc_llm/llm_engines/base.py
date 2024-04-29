from ..samplers import Sampler
from ..chat_templates import ChatTemplate


class LLMEngine:
    """根据"""
    def __init__(
        self,
        sampler: Sampler,
        chat_template: ChatTemplate,
    ):
        self.sampler = sampler
        self.chat_template = chat_template