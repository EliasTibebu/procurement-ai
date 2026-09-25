import json
import os
import threading
import time
import uuid
from typing import Literal, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# =========================================================
# Configuration
# =========================================================

BASE_MODEL = "models/qwen3-0.6b-base"
ADAPTER = "output/procurement-0.6b-lora-v0.2"

MODEL_ID = "procurement-ai"

API_KEY = os.getenv(
    "PROCUREMENT_AI_API_KEY",
    "change-me"
)

torch.set_num_threads(6)
torch.set_num_interop_threads(1)


# =========================================================
# Application
# =========================================================

app = FastAPI(
    title="Procurement AI",
    version="0.3.1"
)


# =========================================================
# Request Models
# =========================================================

class AnalyzeRequest(BaseModel):
    message: str


class ChatMessage(BaseModel):
    role: Literal[
        "system",
        "user",
        "assistant"
    ]
    content: str


class ResponseFormat(BaseModel):
    type: Literal[
        "text",
        "json_object"
    ] = "text"


class ChatCompletionRequest(BaseModel):
    model: str

    messages: list[ChatMessage]

    temperature: float = 0

    max_tokens: Optional[int] = Field(
        default=300,
        ge=1,
        le=1000
    )

    stream: bool = False

    response_format: Optional[ResponseFormat] = None


# =========================================================
# Model Loading
# =========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)


print("Loading procurement LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER
)

model.eval()

print("Procurement model ready.")


# CPU inference should not execute concurrently.
generation_lock = threading.Lock()


# =========================================================
# Authentication
# =========================================================

def verify_api_key(
    authorization: str = "",
    x_api_key: str = ""
):
    """
    Supports:

    Authorization: Bearer <API_KEY>

    and the existing:

    X-API-Key: <API_KEY>
    """

    supplied_key = None

    # OpenAI-compatible Bearer authentication
    if authorization:

        parts = authorization.split(
            " ",
            1
        )

        if (
            len(parts) == 2
            and parts[0].lower() == "bearer"
        ):
            supplied_key = parts[1].strip()

    # Legacy X-API-Key support
    if supplied_key is None and x_api_key:
        supplied_key = x_api_key.strip()

    if supplied_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key.",
                    "type": "authentication_error",
                    "code": "invalid_api_key"
                }
            }
        )


# =========================================================
# Prompt Construction
# =========================================================

def build_procurement_prompt(
    messages: list[ChatMessage]
) -> str:
    """
    Convert OpenAI-compatible messages into the exact
    basic prompt structure used during LoRA training.

    Training format:

    You are a procurement analysis system.
    Return only valid JSON.

    USER:
    <request>

    ASSISTANT:
    """

    user_messages = [
        message.content
        for message in messages
        if message.role == "user"
    ]

    if not user_messages:

        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message":
                        "At least one user message is required.",
                    "type": "invalid_request_error",
                    "code": "missing_user_message"
                }
            }
        )

    # v0.2 is currently trained as a single-turn
    # procurement-analysis model.
    user_message = user_messages[-1]

    return (
        "You are a procurement analysis system.\n"
        "Return only valid JSON.\n\n"
        f"USER:\n{user_message}\n\n"
        "ASSISTANT:\n"
    )


# =========================================================
# Generation
# =========================================================

def generate_response(
    prompt: str,
    max_new_tokens: int = 300
):
    """
    Generate a response using the loaded Qwen base model
    plus procurement LoRA adapter.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    prompt_tokens = inputs["input_ids"].shape[1]

    try:

        # CPU generation should be serialized.
        with generation_lock:

            with torch.inference_mode():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id
                )

        generated = outputs[0][prompt_tokens:]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        completion_tokens = len(generated)

        return (
            response,
            prompt_tokens,
            completion_tokens
        )

    except Exception as error:

        # Log error type server-side without exposing
        # implementation details to the caller.
        print(
            "Model generation failed:",
            type(error).__name__,
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Model generation failed.",
                    "type": "server_error",
                    "code": "generation_failed"
                }
            }
        )


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():

    return {
        "service": "Procurement AI",
        "status": "ok",
        "version": "0.3.1",
        "model": MODEL_ID
    }


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": MODEL_ID,
        "base_model": "qwen3-0.6b",
        "adapter": "procurement-v0.2"
    }


# =========================================================
# OpenAI-Compatible Models
#
# Support BOTH:
#
# GET /models
# GET /v1/models
#
# =========================================================

@app.get("/models")
@app.get("/v1/models")
def list_models(
    authorization: str = Header(default=""),
    x_api_key: str = Header(default="")
):

    verify_api_key(
        authorization=authorization,
        x_api_key=x_api_key
    )

    return {
        "object": "list",

        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "letsprocureai"
            }
        ]
    }


# =========================================================
# OpenAI-Compatible Chat Completions
#
# IMPORTANT:
#
# Support BOTH:
#
# POST /chat/completions
# POST /v1/chat/completions
#
# LetsProcureAI is currently calling:
#
# POST /chat/completions
#
# =========================================================

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(default=""),
    x_api_key: str = Header(default="")
):

    # -----------------------------------------------------
    # Authentication
    # -----------------------------------------------------

    verify_api_key(
        authorization=authorization,
        x_api_key=x_api_key
    )


    # -----------------------------------------------------
    # Model validation
    # -----------------------------------------------------

    if request.model != MODEL_ID:

        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message":
                        f"Model '{request.model}' was not found.",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }
        )


    # -----------------------------------------------------
    # Streaming
    # -----------------------------------------------------

    if request.stream:

        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message":
                        "Streaming is not currently supported.",
                    "type": "invalid_request_error",
                    "code": "streaming_not_supported"
                }
            }
        )


    # -----------------------------------------------------
    # Determine JSON mode
    # -----------------------------------------------------

    json_mode = (
        request.response_format is not None
        and
        request.response_format.type == "json_object"
    )


    # -----------------------------------------------------
    # Build procurement prompt
    # -----------------------------------------------------

    prompt = build_procurement_prompt(
        request.messages
    )


    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    (
        response,
        prompt_tokens,
        completion_tokens
    ) = generate_response(
        prompt=prompt,
        max_new_tokens=request.max_tokens or 300
    )


    # -----------------------------------------------------
    # Validate JSON when requested
    #
    # IMPORTANT:
    # We validate but do NOT modify or repair model output.
    # -----------------------------------------------------

    if json_mode:

        try:

            json.loads(response)

        except json.JSONDecodeError:

            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "message":
                            "The model did not produce valid JSON.",
                        "type": "invalid_model_output",
                        "code": "invalid_json"
                    }
                }
            )


    # -----------------------------------------------------
    # OpenAI-compatible response
    # -----------------------------------------------------

    return {
        "id":
            f"chatcmpl-{uuid.uuid4().hex}",

        "object":
            "chat.completion",

        "created":
            int(time.time()),

        "model":
            MODEL_ID,

        "choices": [
            {
                "index": 0,

                "message": {
                    "role": "assistant",
                    "content": response
                },

                "finish_reason": "stop"
            }
        ],

        "usage": {
            "prompt_tokens":
                prompt_tokens,

            "completion_tokens":
                completion_tokens,

            "total_tokens":
                prompt_tokens + completion_tokens
        }
    }


# =========================================================
# Existing Procurement Analyze Endpoint
#
# Preserved for backward compatibility.
#
# Supports BOTH authentication mechanisms:
#
# X-API-Key
#
# Authorization: Bearer
#
# =========================================================

@app.post("/v1/procurement/analyze")
def analyze(
    request: AnalyzeRequest,
    authorization: str = Header(default=""),
    x_api_key: str = Header(default="")
):

    verify_api_key(
        authorization=authorization,
        x_api_key=x_api_key
    )

    prompt = (
        "You are a procurement analysis system.\n"
        "Return only valid JSON.\n\n"
        f"USER:\n{request.message}\n\n"
        "ASSISTANT:\n"
    )

    (
        response,
        _,
        _
    ) = generate_response(
        prompt=prompt,
        max_new_tokens=300
    )

    try:

        result = json.loads(response)

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_MODEL_OUTPUT",
                "message":
                    "The model did not produce valid JSON."
            }
        )

    return {
        "success": True,
        "model": "procurement-0.6b-v0.2",
        "analysis": result
    }