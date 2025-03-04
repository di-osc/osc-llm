from jsonargparse import CLI
from .chat import main as chat_main
from .servers.openai import main as openai_main
from .quantizers import Int8Quantizer, WeightOnlyInt4Quantizer
from .tokenizer import Tokenizer
from .utils import build_from_checkpoint, get_hf_model_helper
from pathlib import Path
from wasabi import msg
from typing import Literal, Optional
import torch
import os


def download_model(
    repo_id: str,
    save_dir: str = "./checkpoints",
    force_download: bool = False,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    endpoint: Literal["hf", "hf-mirror", "modelscope"] = "hf-mirror",
):
    directory = Path(save_dir, repo_id)

    if endpoint == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=directory,
        )
    else:
        if endpoint == "hf-mirror":
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id,
            local_dir=directory,
            force_download=force_download,
            token=access_token,
        )


def convert(checkpoint_dir: str, save_dir: Optional[str] = None):
    """Convert a huggingface checkpoint to osc_transformers checkpoint.

    Args:
        checkpoint_dir: Path to the directory containing the checkpoint.
        save_dir: Path to the directory to save the converted checkpoint. if None, the converted checkpoint will be saved in the same directory as the original checkpoint.
    """
    model_helper = get_hf_model_helper(checkpoint_dir)
    if not save_dir:
        save_dir = checkpoint_dir
    model_helper.convert_checkpoint(save_dir=save_dir)


def quantize_int8(checkpoint_dir: str, save_dir: str):
    """
    Quantize the osc model to int8.

    Args:
        checkpoint_dir: Path to the osc model directory containing the checkpoint.
        save_dir: Path to the directory to save the quantized model.
    """
    save_dir = Path(save_dir)
    if save_dir == checkpoint_dir:
        msg.warn("The quantized model will replace the original model.")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    model, config = build_from_checkpoint(
        checkpoint_dir=checkpoint_dir, return_config=True
    )
    quantizer = Int8Quantizer()
    model = quantizer.quantize(model)
    config = config.merge(quantizer.quantizer_config)
    torch.save(model.state_dict(), Path(save_dir) / "osc_model.pth")
    config.to_disk(Path(save_dir) / "config.cfg")
    tokenizer.save(save_dir)


def quantize_int4(
    checkpoint_dir: str,
    save_dir: str,
    groupsize: Literal[32, 64, 128, 256] = 32,
    k: Literal[2, 4, 8] = 8,
    padding: bool = True,
    device: str = "cuda:0",
):
    """
    Quantize the osc model to int4.

    Args:
        checkpoint_dir: Path to the osc model directory containing the checkpoint.
        save_dir: Path to the directory to save the quantized model.
        groupsize: The groupsize to use for the quantization.
        k: The k parameter to use for the quantization.
        padding: Whether to pad the model.
        device: The device to use for the quantization.
    """
    save_dir = Path(save_dir)
    if save_dir == checkpoint_dir:
        msg.warn("The quantized model will replace the original model.")
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)
    model, config = build_from_checkpoint(
        checkpoint_dir=checkpoint_dir, return_config=True
    )
    model.to(device)
    quantizer = WeightOnlyInt4Quantizer(
        groupsize=groupsize, inner_k_tiles=k, padding_allowed=padding
    )
    model = quantizer.quantize(model)
    config = config.merge(quantizer.quantizer_config)
    torch.save(model.state_dict(), Path(save_dir) / "osc_model.pth")
    config.to_disk(Path(save_dir) / "config.cfg")
    tokenizer.save(save_dir)


commands = {
    "download": download_model,
    "chat": chat_main,
    "sft": {"lora": lambda: print("lora"), "full": lambda: print("full")},
    "convert": convert,
    "quantize": {"int8": quantize_int8, "int4": quantize_int4},
    "serve": openai_main,
}


def run():
    CLI(components=commands)


if __name__ == "__main__":
    run()
