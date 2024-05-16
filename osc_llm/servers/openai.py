from ..engines import LLMEngineV2, LLMEngineV1, LLMEngine
from ..tokenizer import Tokenizer
from ..chat_templates import Message
from ..utils import random_uuid
from ..samplers import TopK
from typing import List, Optional, Dict, Union, Literal
from pydantic import BaseModel, Field
import torch
import time


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "osc"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


def main(
    checkpoint_dir: str,
    engine: Literal["v1", "v2"] = "v1",
    accelerator: Literal["cuda", "cpu", "gpu", "auto"] = "cuda",
    devices: Union[int, List[int]] = 1,
    max_length: Optional[int] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    compile: bool = True,
):
    """openai接口服务

    Args:
        checkpoint_dir (str): checkpoint目录
        engine (Literal[&#39;v1&#39;, &#39;v2&#39;], optional): LLMEngine版本. Defaults to 'v1'.
        accelerator (Literal[&#39;cuda&#39;, &#39;cpu&#39;, &#39;gpu&#39;, &#39;auto&#39;], optional): 推理硬件. Defaults to 'cuda'.
        devices (Union[int, List[int]], optional): 设备数量或者设备的ID. Defaults to 1.
        host (str, optional): 主机地址. Defaults to '0.0.0.0'.
        port (int, optional): 端口号. Defaults to 8000.
        compile (bool, optional): 是否编译模型. Defaults to True.
    """
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn

    app = FastAPI(description="OpenAI API")

    @app.post("/v1/chat/completions")
    def create_chat_completion(request: ChatCompletionRequest):
        with engine.fabric.init_tensor():
            input_ids = tokenizer.encode_messages(request.messages)
            input_pos = torch.arange(len(input_ids))
        engine.reset_sampler(sampler=TopK(k=100, temperature=request.temperature))
        stream = engine.run(input_ids=input_ids, stop_ids=tokenizer.stop_ids, input_pos=input_pos)
        stream_tokens = tokenizer.decode_stream(stream=stream)

        if request.stream:

            def stream_content(stream_tokens):
                for token in stream_tokens:
                    data = ChatCompletionStreamResponse(
                        id=f"chatcmpl-{random_uuid()}",
                        model=request.model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant", content=token),
                            )
                        ],
                        usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
                    ).model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(content=stream_content(stream_tokens), media_type="text/event-stream")
        else:
            content = ""
            completion_tokens = 0
            for token in stream_tokens:
                content += token
                completion_tokens += 1
            prompt_tokens = len(input_ids)
            total_tokens = prompt_tokens + completion_tokens
            response = ChatCompletionResponse(
                id=f"chatcmpl-{random_uuid()}",
                model=request.model,
                choices=[ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=content))],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens,
                    completion_tokens=completion_tokens,
                ),
            )
            return JSONResponse(content=response.model_dump(exclude_unset=True))

    if engine == "v1":
        engine: LLMEngine = LLMEngineV1(
            checkpoint_dir,
            devices=devices,
            accelerator=accelerator,
            compile=compile,
            max_length=max_length,
        )
    else:
        engine: LLMEngine = LLMEngineV2(
            checkpoint_dir,
            devices=devices,
            accelerator=accelerator,
            compile=compile,
            max_length=max_length,
        )
    engine.setup()
    tokenizer = Tokenizer(checkpoint_dir=checkpoint_dir)

    # Todo: 在启动模型编译的情况下第一次运行需要耗费很多时间(几分钟),如何在启动的时候预热模型?
    if compile:
        from wasabi import msg

        msg.warn("you are using compile mode, the first run may take a long time")
    uvicorn.run(app=app, host=host, port=port)
