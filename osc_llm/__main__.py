from jsonargparse import CLI

from osc_llm import LLM


def chat(
    checkpoint_dir: str,
    gpu_memory_utilization: float = 0.5,
    device: str = "cuda",
    enable_thinking: bool = True,
):
    """chat with a model

    Args:
        checkpoint_dir (str): the path to the checkpoint directory
        gpu_memory_utilization (float, optional): the GPU memory utilization. Defaults to 0.5.
        device (str, optional): the device to use. Defaults to "cuda".
    """
    llm = LLM(checkpoint_dir, gpu_memory_utilization, device)
    messages = []
    while True:
        user_input = input("Input: ")
        if user_input == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        answer = ""
        for token in llm.chat(messages, enable_thinking=enable_thinking, stream=True):
            answer += token
            print(token, end="", flush=True)
        messages.append({"role": "assistant", "content": answer})
        print()


commands = {
    "chat": chat,
}


def run_cli():
    CLI(components=commands)


if __name__ == "__main__":
    run_cli()
