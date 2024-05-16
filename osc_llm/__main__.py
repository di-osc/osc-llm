from jsonargparse import CLI
from .chat import main as chat_main
from .servers.openai import main as openai_main
from .model_helpers import get_supported_architectures
from .model_helpers.base import HFModelHelper
from .quantizers import Int8Quantizer, WeightOnlyInt4Quantizer
from .tokenizer import Tokenizer
from .config import registry
from .utils import build_from_checkpoint
from pathlib import Path
from wasabi import msg
from typing import Literal, Optional
import json
import torch
from lightning_utilities.core.imports import RequirementCache
import os


_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")


def download_huggingface_model(
    repo_id: str,
    save_dir: str = "./checkpoints",
    force_download: bool = True,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    from_safetensors: bool = False,
):
    directory = Path(save_dir, repo_id)
    if directory.exists():
        if not force_download:
            msg.fail(
                f"Directory {directory} already exists. Use --force_download to re-download.",
                exits=1,
            )
        else:
            msg.info(f"Directory {directory} already exists. Re-downloading.")
            import shutil

            shutil.rmtree(directory)
    if not directory.exists():
        directory.mkdir(parents=True)
    from huggingface_hub import snapshot_download

    download_files = ["tokenizer*", "generation_config.json", "config.json"]
    if from_safetensors:
        if not _SAFETENSORS_AVAILABLE:
            raise ModuleNotFoundError(str(_SAFETENSORS_AVAILABLE))
        download_files.append("*.safetensors")
    else:
        download_files.append("*.bin*")
    msg.info(f"Saving to {directory}")
    snapshot_download(
        repo_id,
        local_dir=directory,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=download_files,
        token=access_token,
    )

    # convert safetensors to PyTorch binaries
    if from_safetensors:
        from safetensors import SafetensorError
        from safetensors.torch import load_file as safetensors_load
        import torch

        msg.info("Converting .safetensor files to PyTorch binaries (.bin)")
        for safetensor_path in directory.glob("*.safetensors"):
            bin_path = safetensor_path.with_suffix(".bin")
            try:
                result = safetensors_load(safetensor_path)
            except SafetensorError as e:
                raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
            msg.info(f"{safetensor_path} --> {bin_path}")
            torch.save(result, bin_path)
            os.remove(safetensor_path)


def get_hf_model_helper(checkpoint_dir: str) -> HFModelHelper:
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    architecture = config["architectures"][0]
    allowed_architectures = get_supported_architectures()
    if architecture not in allowed_architectures:
        msg.fail(
            title="Architecture {architecture} is not supported.",
            text=f"Supported architectures are: {allowed_architectures}",
            exits=1,
        )
    model_helper: HFModelHelper = registry.model_helpers.get(architecture)(checkpoint_dir)
    return model_helper


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
    model, config = build_from_checkpoint(checkpoint_dir=checkpoint_dir, return_config=True)
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
    model, config = build_from_checkpoint(checkpoint_dir=checkpoint_dir, return_config=True)
    model.to(device)
    quantizer = WeightOnlyInt4Quantizer(groupsize=groupsize, inner_k_tiles=k, padding_allowed=padding)
    model = quantizer.quantize(model)
    config = config.merge(quantizer.quantizer_config)
    torch.save(model.state_dict(), Path(save_dir) / "osc_model.pth")
    config.to_disk(Path(save_dir) / "config.cfg")
    tokenizer.save(save_dir)


commands = {
    "download": download_huggingface_model,
    "chat": chat_main,
    "sft": {"lora": lambda: print("lora"), "full": lambda: print("full")},
    "convert": convert,
    "quantize": {"int8": quantize_int8, "int4": quantize_int4},
    "serve": openai_main,
}


def run():
    CLI(components=commands, as_positional=False)


if __name__ == "__main__":
    run()
