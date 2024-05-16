import json
from pathlib import Path
from typing import Optional, Union, Generator, List
import torch
from .chat_templates import ChatTemplate, Message
from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(
        self,
        checkpoint_dir: Union[Path, str],
        chat_template: Optional[ChatTemplate] = None,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        self.chat_template = chat_template if chat_template else ChatTemplate.from_checkpoint(checkpoint_dir)

        # some checkpoints have both files, `.model` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            self.tokenizer_path = checkpoint_dir / "tokenizer.model"
            self.processor: SentencePieceProcessor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()

        elif (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            self.tokenizer_path = checkpoint_dir / "tokenizer.json"
            self.processor: HFTokenizer = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                self.tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                eos_token = config.get("eos_token")
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                self.generation_config_path = checkpoint_dir / "generation_config.json"
                with open(special_tokens_path) as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
        else:
            raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path) as fp:
            config = json.load(fp)
        if any(config.get(check, False) for check in ("add_bos_token", "add_prefix_space")):
            return True
        # for examples that also use the Llama tokenizer, but do not have or set add_bos_token to True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("add_bos_token") is None and config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            self.processor: HFTokenizer
            tokens = self.processor.encode(string, add_special_tokens=False).ids
        elif self.backend == "sentencepiece":
            self.processor: SentencePieceProcessor
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError
        if bos or (bos is None and self.use_bos):
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def encode_messages(
        self,
        messages: List[Message],
        add_generate_prompt: bool = True,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        assert self.chat_template, "Chat template is required for encoding messages"
        string = self.chat_template.apply_messages(messages, add_generate_prompt=add_generate_prompt)
        return self.encode(string, device, bos, eos, max_length)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)

    def decode_stream(
        self,
        stream: Generator[torch.Tensor, None, None],
    ) -> Generator[str, None, None]:
        if self.backend == "huggingface":
            buffer = torch.tensor([], dtype=torch.long)
            text = ""
            try:
                for token in stream:
                    buffer = buffer.to(device=token.device)
                    buffer = torch.cat((buffer, token.view(-1)))
                    t = self.decode(buffer)
                    if not self.has_special_chars(t):
                        yield t
                        text += t
                        buffer = torch.tensor([], dtype=torch.long)
            except KeyboardInterrupt:
                # support stopping generation
                return text
        elif self.backend == "sentencepiece":
            # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
            # meaning that we need to decode everything each time
            so_far = torch.tensor([], dtype=torch.long)
            decoded_so_far = ""
            try:
                for token in stream:
                    so_far = so_far.to(device=token.device)
                    so_far = torch.cat((so_far, token.view(-1)))
                    decoded_new = self.decode(so_far)
                    if self.has_special_chars(decoded_new):
                        # if the text contains special characters, it means that the tokenization is not complete
                        continue
                    yield decoded_new[len(decoded_so_far) :]
                    decoded_so_far = decoded_new
            except KeyboardInterrupt:
                # support stopping generation
                return decoded_so_far
        else:
            raise NotImplementedError(self.backend)

    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        if save_dir == self.checkpoint_dir:
            return
        import shutil

        if self.backend == "huggingface":
            shutil.copyfile(self.tokenizer_path, save_dir / self.tokenizer_path.name)
            shutil.copyfile(self.tokenizer_config_path, save_dir / self.tokenizer_config_path.name)
            shutil.copyfile(self.generation_config_path, save_dir / self.generation_config_path.name)
        if self.backend == "sentencepiece":
            shutil.copyfile(self.tokenizer_path, save_dir / self.tokenizer_path.name)

    @property
    def stop_ids(self) -> List[List[int]]:
        stop_ids = [torch.tensor([self.eos_id], dtype=torch.int)]
        if self.chat_template:
            stop_ids.extend([self.encode(text) for text in self.chat_template.stop_texts])
        return stop_ids

    def has_special_chars(self, text: str) -> bool:
        """使用sentencepiece时，检查文本中是否包含特殊字符�.这种情况通常是由于一个中文字符被分割为几个token,而解码时没有合并回去导致的."""
        return "�" in text
