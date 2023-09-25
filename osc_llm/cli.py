from typer import Typer
import os
from lightning_utilities.core.imports import RequirementCache
import typer
import contextlib
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from .utils import NotYetLoadedTensor, incremental_save, lazy_load
from .llm import LlamaConfig
from enum import Enum
import torch


app = Typer()

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")

@app.command('download')
def download_huggingface_model(repo_id: str, 
                               save_dir: str = "./checkpoints",
                               force_download: bool = False,
                               access_token: Optional[str] = os.getenv("HF_TOKEN"),
                               from_safetensors: bool = False):
    
    directory = Path(save_dir, repo_id)
    if directory.exists():
        if not force_download:
            typer.echo(f"Directory {directory} already exists. Use --force-download to re-download.")
            return
        else:
            typer.echo(f"Directory {directory} already exists. Re-downloading.")
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
    typer.echo(f"Saving to {directory}")
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

        print("Converting .safetensor files to PyTorch binaries (.bin)")
        for safetensor_path in directory.glob("*.safetensors"):
            bin_path = safetensor_path.with_suffix(".bin")
            try:
                result = safetensors_load(safetensor_path)
            except SafetensorError as e:
                raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
            print(f"{safetensor_path} --> {bin_path}")
            torch.save(result, bin_path)
            os.remove(safetensor_path)
            
class Accelerator(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    
class Quantize(str, Enum):
    bnb_nf4 = "bnb.nf4"
    bnb_nf4_dq = "bnb.nf4-dq"
    bnb_fp4 = "bnb.fp4"
    bnb_fp4_dq = "bnb.fp4-dq"
    bnb_int8 = "bnb.int8"
    gptq_int4 = "gptq.int4"
    
@app.command('chat')
def chat_with_model(checkpoint_dir: Path,
                    accelerator: Accelerator = Accelerator.cuda,
                    devices: str = '0',
                    top_k: int = 200,
                    temperature: float = 0.8,
                    quantize: Optional[Quantize] = None,
                    precision: Optional[str] = None,):
    from .chat import main
    if ',' in devices:
        devices = [int(d) for d in devices.split(',')]
    elif '[' in devices:
        devices = [int(d) for d in devices[1:-1].split(',')]
    else:
        devices = int(devices)
    main(checkpoint_dir=checkpoint_dir, 
         accelerator=accelerator.value,
         devices=devices,
         top_k=top_k, 
         temperature=temperature, 
         quantize=quantize, 
         precision=precision)


@app.command('convert')
def convert_hf(source_dir: Path, 
               size: str = "7B", 
               dtype: Optional[str] = None):
    
    def copy_weights_hf_llama(state_dict: Dict[str, torch.Tensor],
                              hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
                              saver: Optional[incremental_save] = None,
                              dtype: Optional[torch.dtype] = None) -> None:
        weight_map = {
            "model.embed_tokens.weight": "transformer.token_embeddings.weight",
            "model.layers.{}.input_layernorm.weight": "transformer.blocks.{}.norm_attn.weight",
            "model.layers.{}.self_attn.q_proj.weight": "transformer.blocks.{}.attn.q_proj.weight",
            "model.layers.{}.self_attn.k_proj.weight": "transformer.blocks.{}.attn.k_proj.weight",
            "model.layers.{}.self_attn.v_proj.weight": "transformer.blocks.{}.attn.v_proj.weight",
            "model.layers.{}.self_attn.o_proj.weight": "transformer.blocks.{}.attn.o_proj.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.post_attention_layernorm.weight": "transformer.blocks.{}.norm_mlp.weight",
            "model.layers.{}.mlp.gate_proj.weight": "transformer.blocks.{}.mlp.fc_1.weight",
            "model.layers.{}.mlp.up_proj.weight": "transformer.blocks.{}.mlp.fc_2.weight",
            "model.layers.{}.mlp.down_proj.weight": "transformer.blocks.{}.mlp.proj.weight",
            "model.norm.weight": "transformer.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        for name, param in hf_weights.items():
            if "model.layers" in name:
                from_name, number = layer_template(name, 2)
                to_name: str = weight_map[from_name]
                if to_name is None:
                    continue
                to_name = to_name.format(number)
            else:
                to_name = weight_map[name]
            param = load_param(param, name, dtype)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param



    def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
        split = layer_name.split(".")
        number = int(split[idx])
        split[idx] = "{}"
        from_name = ".".join(split)
        return from_name, number


    def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
        if hasattr(param, "_load_tensor"):
            # support tensors loaded via `lazy_load()`
            print(f"Loading {name!r} into RAM")
            param = param._load_tensor()
        if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
            param = param.to(dtype)
        return param

    with torch.inference_mode():
        
        if dtype is not None:
            dtype = getattr(torch, dtype)
        config = LlamaConfig.from_name(size)
        config_dict = asdict(config)
        print(f"Model config {config_dict}")
        with open(source_dir / "llm_config.json", "w") as json_config:
            json.dump(config_dict, json_config)

        copy_fn = copy_weights_hf_llama

        # initialize a new empty state dict to hold our new weights
        sd = {}

        # Load the json file containing weight mapping
        pytorch_bin_map_json_path = source_dir / "pytorch_model.bin.index.json"
        if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
            with open(pytorch_bin_map_json_path) as json_map:
                bin_index = json.load(json_map)
            bin_files = {source_dir / bin for bin in bin_index["weight_map"].values()}
        else:
            bin_files = set(source_dir.glob("*.bin"))
            # some checkpoints serialize the training arguments
            bin_files = {f for f in bin_files if f.name != "training_args.bin"}
        if not bin_files:
            raise ValueError(f"Expected {str(source_dir)!r} to contain .bin files")

        with incremental_save(source_dir / "llm.pth") as saver:
            # for checkpoints that split the QKV across several files, we need to keep all the bin files
            # open, so we use `ExitStack` to close them all together at the end
            with contextlib.ExitStack() as stack:
                for bin_file in sorted(bin_files):
                    print("Processing", bin_file)
                    hf_weights = stack.enter_context(lazy_load(bin_file))
                    copy_fn(sd, hf_weights, saver=saver, dtype=dtype)
                gc.collect()
            print("Saving converted checkpoint")
            saver.save(sd)
if __name__ == "__main__":
    app()