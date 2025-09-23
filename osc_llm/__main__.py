import os
from pathlib import Path
from typing import Literal, Optional

from jsonargparse import CLI


def download_model(
    repo_id: str,
    save_dir: str = "./checkpoints",
    force_download: bool = False,
    access_token: Optional[str] = os.getenv("HF_TOKEN"),
    endpoint: Literal["hf", "hf-mirror", "modelscope"] = "hf-mirror",
):
    """
    Download a model from the Hugging Face Hub or ModelScope.

    Args:
        repo_id: The ID of the model to download.
        save_dir: The directory to save the downloaded model.
        force_download: Whether to force the download of the model.
        access_token: The access token to use for the Hugging Face Hub.
        endpoint: The endpoint to use for the Hugging Face Hub.
    """
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


def serve_openai(
    model_path: str,
    port: int = 8000,
):
    """
    Serve the osc model as an OpenAI API.

    Args:
        model_path: Path to the osc model directory containing the checkpoint.
        port: The port to serve the OpenAI API.
    """
    from .servers.openai import main as openai_main

    openai_main(model_path, port)


commands = {
    "download": download_model,
    "serve": serve_openai,
}


def run_cli():
    CLI(components=commands)


if __name__ == "__main__":
    run_cli()
