#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import uuid
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


DEBUG_MODE = str(
    os.getenv(
        "DEBUG_MODE",
        "false",
    )
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}

POLICY_SAMPLER_MODEL = os.getenv(
    "POLICY_SAMPLER_MODEL",
    "Qwen2-VL-7B-Instruct",
)

llamafactory_src = os.getenv(
    "LLAMAFACTORY_SRC"
)

if llamafactory_src:
    sys.path.append(
        llamafactory_src
    )

from llamafactory.chat.chat_model import ChatModel


_SAMPLER = None


def _debug(
    message: str,
) -> None:
    if DEBUG_MODE:
        print(
            message,
            flush=True,
        )


def _build_sampler(
    model_path: str,
) -> ChatModel:
    args = {
        "model_name_or_path": model_path,
        "infer_backend": "huggingface",
        "template": "qwen2_vl",
        "trust_remote_code": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 256,
        "num_beams": 1,
    }

    _debug(
        "[SAMPLER] Loading model."
    )

    sampler = ChatModel(
        args=args
    )

    _debug(
        "[SAMPLER] Model ready."
    )

    return sampler


def get_sampler() -> ChatModel:
    global _SAMPLER

    if _SAMPLER is None:
        _SAMPLER = _build_sampler(
            POLICY_SAMPLER_MODEL
        )

    return _SAMPLER


def _stream_once(
    chat_model,
    messages,
    system,
    images,
    **gen_args,
) -> str:
    chunks = []

    for token in chat_model.stream_chat(
        messages=messages,
        system=system,
        images=images,
        **gen_args,
    ):
        chunks.append(
            token
        )

    return "".join(
        chunks
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    image_path: str


class ChatCompletionChoiceMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[
        ChatCompletionChoice
    ]
    usage: ChatCompletionUsage


app = FastAPI(
    title="Qwen2-VL Sampler",
    version="0.1.0",
)


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
def chat_completions(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if not req.messages:
        raise HTTPException(
            400,
            "messages must be non-empty",
        )

    prompt = (
        req.messages[-1].content
        or ""
    )

    image_path = req.image_path

    if not image_path:
        raise HTTPException(
            400,
            "image_path is required",
        )

    if DEBUG_MODE:
        _debug(
            f"[SAMPLER] Request received: "
            f"prompt_chars={len(prompt)} "
            f"image={os.path.basename(image_path)}"
        )

    sampler = get_sampler()

    temperature = (
        req.temperature
        if req.temperature is not None
        else 0.4
    )

    max_new_tokens = (
        req.max_tokens
        if req.max_tokens is not None
        else 256
    )

    top_p = (
        req.top_p
        if req.top_p is not None
        else 0.9
    )

    try:
        text = _stream_once(
            sampler,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            system="",
            images=[
                image_path
            ],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
        ).strip()

    except Exception as exc:
        _debug(
            f"[SAMPLER] Generation failed: "
            f"{type(exc).__name__}"
        )

        raise HTTPException(
            500,
            "Sampler generation failed.",
        ) from None

    if DEBUG_MODE:
        _debug(
            f"[SAMPLER] Response generated: "
            f"chars={len(text)}"
        )

    choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionChoiceMessage(
            role="assistant",
            content=text,
        ),
        finish_reason="stop",
    )

    return ChatCompletionResponse(
        id=(
            "chatcmpl-"
            + uuid.uuid4().hex[:8]
        ),
        object="chat.completion",
        created=int(
            time.time()
        ),
        model=req.model,
        choices=[
            choice
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv(
        "SAMPLER_HOST",
        "127.0.0.1",
    )

    port = int(
        os.getenv(
            "SAMPLER_PORT",
            "9000",
        )
    )

    if DEBUG_MODE:
        _debug(
            "[SAMPLER] Debug mode enabled."
        )

        _debug(
            f"[SAMPLER] Listening on "
            f"{host}:{port}"
        )

    uvicorn.run(
        app,
        host=host,
        port=port,
    )