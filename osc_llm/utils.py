from pathlib import Path
from .config import registry, Config
from typing import Optional, Union, Dict, Tuple
from wasabi import msg
import statistics
import torch
import uuid


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_chat_template(name) -> Optional[Config]:
    """在一个checkpoint的名称中获取chat模板"""
    for k, v in registry.chat_templates.get_all().items():
        if k in name:
            return v
    return None


def build_model(
    config: Union[Dict, str, Path, Config],
    model_section: str = "model",
    quantizer_section: str = "quantizer",
    empty_init: bool = True,
    quantize: bool = True,
    return_config: bool = False,
) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Config]]:
    """Build a model from a configuration.

    Args:
        config (Union[Dict, str, Path, Config]): the configuration to build the model from, can be a dictionary, a path to a file or a Config object.
        model_section (str, optional): the section to look for the model in the configuration. Defaults to 'model'.
        quantizer_section (str, optional): the section to look for the quantizer in the configuration. Defaults to 'quantizer'.
        empty_init (bool, optional): whether to initialize the model with empty weights. Defaults to True.
        quantize (bool, optional): whether to quantize the model. Defaults to True.
        return_config (bool, optional): whether to return the configuration as well. Defaults to False.

    Returns:
        torch.nn.Module: the model built from the configuration.
    """
    if isinstance(config, (str, Path)):
        config = Config().from_disk(config)
    if isinstance(config, dict):
        config = Config(data=config)
    if empty_init:
        with torch.device("meta"):
            resolved = registry.resolve(config=config)
    else:
        resolved = registry.resolve(config=config)
    if model_section not in resolved:
        msg.fail(f"cannot find model section {model_section}")
    else:
        model = resolved[model_section]
    if quantizer_section in resolved and quantize:
        quantizer = resolved[quantizer_section]
        model = quantizer.convert_for_runtime(model=model)
    if return_config:
        return model, config
    return model


def build_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    model_section: str = "model",
    config_name: str = "config.cfg",
    model_name: str = "osc_model.pth",
    empty_init: bool = True,
    quantize: bool = True,
    weights_only: bool = True,
    load_weights: bool = True,
    return_config: bool = False,
):
    """build a model from a checkpoint directory.

    Args:
        checkpoint_dir (Union[str, Path]): the directory containing the model checkpoint.
        model_section (str, optional): the section to look for the model in the configuration. Defaults to 'model'.
        config_name (str, optional): the name of the configuration file. Defaults to 'config.cfg'.
        model_name (str, optional): the name of the model file. Defaults to 'osc_model.pth'.
        empty_init (bool, optional): whether to initialize the model with empty weights which means that the model will be built on meta device. Defaults to True.
        quantize (bool, optional): whether to convert the model to a quantized model. Defaults to True.
        weights_only (bool, optional): whether to load only the weights from the checkpoint. Defaults to True.
        load_weights (bool, optional): whether to load the weights from the checkpoint. Defaults to True.
        return_config (bool, optional): whether to return the configuration as well. Defaults to False.

    Returns:
        torch.nn.Module: the model loaded from the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    config_path = Path(checkpoint_dir) / config_name
    if return_config:
        model, config = build_model(
            config_path,
            model_section=model_section,
            quantize=quantize,
            empty_init=empty_init,
            return_config=return_config,
        )
    else:
        model = build_model(
            config_path,
            model_section=model_section,
            quantize=quantize,
            empty_init=empty_init,
            return_config=return_config,
        )
    if load_weights:
        states = torch.load(
            str(checkpoint_dir / model_name),
            map_location="cpu",
            mmap=True,
            weights_only=weights_only,
        )
        model.load_state_dict(states)
    if return_config:
        return model, config
    return model


@torch.no_grad()
def benchmark(model, num_iters=10, **inputs):
    """Runs the model on the input several times and returns the median execution time."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        start.record()
        model(**inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    msg.info(f"Number of iterations: {num_iters}.")
    all_time = sum(times)
    msg.info(f"Total time: {all_time:.4f} s")
    mean_time = statistics.mean(times)
    msg.info(f"Mean time: {mean_time:.4f} s")
    median_time = statistics.median(times)
    msg.info(f"Median time: {median_time:.4f} s")


def get_default_supported_precision(training: bool) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def get_model_size(model: torch.nn.Module, contains_embedding: bool = False) -> int:
    """Get the size of a model in bytes.

    Args:
        model (torch.nn.Module): the model to get the size of.
        contains_embedding (bool, optional): whether the model contains an embedding layer. Defaults to False.

    Returns:
        int: the size of the model in bytes.
    """
    size = 0
    for param in model.parameters():
        size += param.numel() * param.element_size()
    if contains_embedding:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                size += module.num_embeddings * module.embedding_dim * module.weight.element_size()
    return size
