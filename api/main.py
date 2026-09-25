import json
import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Literal, Optional

import torch
from fastapi import FastAPI, Header, HTTPException, Request
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
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )
)

logger = logging.getLogger("procurement-ai")

# Stores the current request ID so helper functions can log
# against the same request without passing request_id around.
request_id_context = ContextVar(
    "request_id",
    default="-"
)


def log(message: str):
    """
    Write an application log associated with the current
    HTTP request.
    """

    request_id = request_id_context.get()

    logger.info(
        "[request_id=%s] %s",
        request_id,
        message
    )


# =========================================================
# Application
# =========================================================

app = FastAPI(
    title="Procurement AI",
    version="0.4.0"
)


# =========================================================
# HTTP Request Logging Middleware
# =========================================================

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next
):
    """
    Logs the request from the moment it reaches FastAPI
    until the HTTP response is returned.

    IMPORTANT:
    Authorization headers and API keys are NEVER logged.
    """

    request_id = uuid.uuid4().hex[:12]

    context_token = request_id_context.set(
        request_id
    )

    started = time.perf_counter()

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    logger.info(
        "[request_id=%s] --> REQUEST START | "
        "method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        client_ip
    )

    try:

        response = await call_next(request)

        elapsed = (
            time.perf_counter()
            - started
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        logger.info(
            "[request_id=%s] <-- REQUEST END | "
            "status=%s duration=%.2fs",
            request_id,
            response.status_code,
            elapsed
        )

        return response

    except Exception:

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.exception(
            "[request_id=%s] !! REQUEST FAILED | "
            "duration=%.2fs",
            request_id,
            elapsed
        )

        raise

    finally:

        request_id_context.reset(
            context_token
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

    response_format: Optional[
        ResponseFormat
    ] = None


# =========================================================
# Model Loading
# =========================================================

logger.info(
    "MODEL | Loading tokenizer | path=%s",
    BASE_MODEL
)

model_load_started = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


logger.info(
    "MODEL | Tokenizer loaded"
)


logger.info(
    "MODEL | Loading base model | path=%s",
    BASE_MODEL
)

base_load_started = time.perf_counter()

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

base_load_time = (
    time.perf_counter()
    - base_load_started
)

logger.info(
    "MODEL | Base model loaded | duration=%.2fs",
    base_load_time
)


logger.info(
    "MODEL | Loading procurement LoRA | path=%s",
    ADAPTER
)

adapter_load_started = time.perf_counter()

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER
)

adapter_load_time = (
    time.perf_counter()
    - adapter_load_started
)

model.eval()

total_model_load_time = (
    time.perf_counter()
    - model_load_started
)

logger.info(
    "MODEL | Procurement LoRA loaded | duration=%.2fs",
    adapter_load_time
)

logger.info(
    "MODEL | READY | model=%s total_load_time=%.2fs",
    MODEL_ID,
    total_model_load_time
)


# =========================================================
# Generation Lock
# =========================================================

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

    and:

    X-API-Key: <API_KEY>

    The actual API key is NEVER logged.
    """

    log(
        "AUTH | Starting API key validation"
    )

    supplied_key = None
    auth_method = "none"

    # -----------------------------------------------------
    # OpenAI-compatible Bearer authentication
    # -----------------------------------------------------

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
            auth_method = "bearer"

    # -----------------------------------------------------
    # Legacy X-API-Key
    # -----------------------------------------------------

    if supplied_key is None and x_api_key:

        supplied_key = x_api_key.strip()
        auth_method = "x-api-key"

    log(
        f"AUTH | Credential received | "
        f"method={auth_method}"
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if supplied_key != API_KEY:

        log(
            f"AUTH | REJECTED | "
            f"method={auth_method}"
        )

        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message":
                        "Invalid API key.",
                    "type":
                        "authentication_error",
                    "code":
                        "invalid_api_key"
                }
            }
        )

    log(
        f"AUTH | SUCCESS | "
        f"method={auth_method}"
    )


# =========================================================
# Prompt Construction
# =========================================================

def build_procurement_prompt(
    messages: list[ChatMessage]
) -> str:
    """
    Converts OpenAI-compatible messages into the prompt
    format used when training procurement LoRA v0.2.

    Training format:

    You are a procurement analysis system.
    Return only valid JSON.

    USER:
    <request>

    ASSISTANT:
    """

    log(
        f"PROMPT | Building procurement prompt | "
        f"messages={len(messages)}"
    )

    user_messages = [
        message.content
        for message in messages
        if message.role == "user"
    ]

    if not user_messages:

        log(
            "PROMPT | FAILED | "
            "No user message found"
        )

        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message":
                        "At least one user message is required.",
                    "type":
                        "invalid_request_error",
                    "code":
                        "missing_user_message"
                }
            }
        )

    # -----------------------------------------------------
    # Current v0.2 model is trained as a single-turn
    # procurement analysis model.
    #
    # Therefore use the latest user message.
    # -----------------------------------------------------

    user_message = user_messages[-1]

    log(
        f"PROMPT | User message selected | "
        f"characters={len(user_message)}"
    )

    prompt = (
        "You are a procurement analysis system.\n"
        "Return only valid JSON.\n\n"
        f"USER:\n{user_message}\n\n"
        "ASSISTANT:\n"
    )

    log(
        f"PROMPT | READY | "
        f"characters={len(prompt)}"
    )

    return prompt


# =========================================================
# Model Generation
# =========================================================

def generate_response(
    prompt: str,
    max_new_tokens: int = 300
):
    """
    Generate a response using:

    Qwen3-0.6B
        +
    procurement LoRA v0.2

    Logs:
    - tokenization
    - prompt tokens
    - inference queue
    - lock wait
    - inference duration
    - completion tokens
    - tokens/second
    - total generation duration
    """

    generation_started = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Tokenization
    # -----------------------------------------------------

    log(
        f"GENERATION | Tokenization started | "
        f"max_new_tokens={max_new_tokens}"
    )

    tokenization_started = (
        time.perf_counter()
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    tokenization_time = (
        time.perf_counter()
        - tokenization_started
    )

    prompt_tokens = (
        inputs["input_ids"].shape[1]
    )

    log(
        f"GENERATION | Tokenization complete | "
        f"prompt_tokens={prompt_tokens} "
        f"duration={tokenization_time:.4f}s"
    )

    try:

        # -------------------------------------------------
        # Wait for CPU inference lock
        # -------------------------------------------------

        log(
            "GENERATION | Waiting for inference lock"
        )

        lock_wait_started = (
            time.perf_counter()
        )

        with generation_lock:

            lock_wait_time = (
                time.perf_counter()
                - lock_wait_started
            )

            log(
                f"GENERATION | Lock acquired | "
                f"wait={lock_wait_time:.2f}s"
            )

            # ---------------------------------------------
            # Inference
            # ---------------------------------------------

            inference_started = (
                time.perf_counter()
            )

            log(
                "GENERATION | Model inference started"
            )

            with torch.inference_mode():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id
                )

            inference_time = (
                time.perf_counter()
                - inference_started
            )

            log(
                f"GENERATION | Model inference finished | "
                f"duration={inference_time:.2f}s"
            )

        # -------------------------------------------------
        # Extract generated tokens
        # -------------------------------------------------

        generated = outputs[0][
            prompt_tokens:
        ]

        completion_tokens = len(
            generated
        )

        log(
            f"GENERATION | Output received | "
            f"completion_tokens={completion_tokens}"
        )

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------

        decoding_started = (
            time.perf_counter()
        )

        log(
            "GENERATION | Decoding started"
        )

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        decoding_time = (
            time.perf_counter()
            - decoding_started
        )

        log(
            f"GENERATION | Decoding complete | "
            f"duration={decoding_time:.4f}s "
            f"response_chars={len(response)}"
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        total_time = (
            time.perf_counter()
            - generation_started
        )

        tokens_per_second = (
            completion_tokens
            / inference_time
            if inference_time > 0
            else 0
        )

        log(
            f"GENERATION | COMPLETE | "
            f"prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} "
            f"lock_wait={lock_wait_time:.2f}s "
            f"inference={inference_time:.2f}s "
            f"total={total_time:.2f}s "
            f"speed={tokens_per_second:.2f}tok/s"
        )

        return (
            response,
            prompt_tokens,
            completion_tokens
        )

    except HTTPException:
        raise

    except Exception as error:

        logger.exception(
            "[request_id=%s] "
            "GENERATION | FAILED | "
            "error_type=%s",
            request_id_context.get(),
            type(error).__name__
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message":
                        "Model generation failed.",
                    "type":
                        "server_error",
                    "code":
                        "generation_failed"
                }
            }
        )


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():

    log(
        "ROOT | Service information requested"
    )

    return {
        "service":
            "Procurement AI",

        "status":
            "ok",

        "version":
            "0.4.0",

        "model":
            MODEL_ID
    }


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    log(
        "HEALTH | Health check requested"
    )

    return {
        "status":
            "ok",

        "model":
            MODEL_ID,

        "base_model":
            "qwen3-0.6b",

        "adapter":
            "procurement-v0.2"
    }


# =========================================================
# OpenAI-Compatible Models
#
# Supports:
#
# GET /models
# GET /v1/models
#
# =========================================================

@app.get("/models")
@app.get("/v1/models")
def list_models(
    authorization: str = Header(
        default=""
    ),
    x_api_key: str = Header(
        default=""
    )
):

    log(
        "MODELS | Model list requested"
    )

    verify_api_key(
        authorization=authorization,
        x_api_key=x_api_key
    )

    log(
        f"MODELS | Returning model | "
        f"model={MODEL_ID}"
    )

    return {
        "object": "list",

        "data": [
            {
                "id":
                    MODEL_ID,

                "object":
                    "model",

                "created":
                    0,

                "owned_by":
                    "letsprocureai"
            }
        ]
    }


# =========================================================
# OpenAI-Compatible Chat Completions
#
# Supports BOTH:
#
# POST /chat/completions
# POST /v1/chat/completions
#
# =========================================================

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,

    authorization: str = Header(
        default=""
    ),

    x_api_key: str = Header(
        default=""
    )
):

    chat_started = (
        time.perf_counter()
    )

    # -----------------------------------------------------
    # Request parsed
    # -----------------------------------------------------

    log(
        f"CHAT | Request parsed | "
        f"model={request.model} "
        f"messages={len(request.messages)} "
        f"stream={request.stream} "
        f"temperature={request.temperature} "
        f"max_tokens={request.max_tokens}"
    )

    # -----------------------------------------------------
    # Authentication
    # -----------------------------------------------------

    log(
        "CHAT | Starting authentication"
    )

    verify_api_key(
        authorization=authorization,
        x_api_key=x_api_key
    )

    log(
        "CHAT | Authentication passed"
    )

    # -----------------------------------------------------
    # Model validation
    # -----------------------------------------------------

    log(
        f"CHAT | Validating model | "
        f"requested={request.model}"
    )

    if request.model != MODEL_ID:

        log(
            f"CHAT | Model validation FAILED | "
            f"requested={request.model} "
            f"available={MODEL_ID}"
        )

        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message":
                        f"Model '{request.model}' "
                        f"was not found.",

                    "type":
                        "invalid_request_error",

                    "code":
                        "model_not_found"
                }
            }
        )

    log(
        "CHAT | Model validation successful"
    )

    # -----------------------------------------------------
    # Streaming
    # -----------------------------------------------------

    if request.stream:

        log(
            "CHAT | Streaming requested | "
            "status=unsupported"
        )

        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message":
                        "Streaming is not currently supported.",

                    "type":
                        "invalid_request_error",

                    "code":
                        "streaming_not_supported"
                }
            }
        )

    # -----------------------------------------------------
    # Response format
    # -----------------------------------------------------

    json_mode = (
        request.response_format is not None
        and
        request.response_format.type
        == "json_object"
    )

    log(
        f"CHAT | Response format determined | "
        f"json_mode={json_mode}"
    )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    log(
        "CHAT | Building procurement prompt"
    )

    prompt = build_procurement_prompt(
        request.messages
    )

    log(
        "CHAT | Procurement prompt ready"
    )

    # -----------------------------------------------------
    # Model inference
    # -----------------------------------------------------

    log(
        "CHAT | Sending request to model"
    )

    (
        response,
        prompt_tokens,
        completion_tokens
    ) = generate_response(
        prompt=prompt,
        max_new_tokens=(
            request.max_tokens
            or 300
        )
    )

    log(
        f"CHAT | Model response received | "
        f"completion_tokens={completion_tokens}"
    )

    # -----------------------------------------------------
    # JSON validation
    # -----------------------------------------------------

    if json_mode:

        log(
            "CHAT | JSON validation started"
        )

        try:

            json.loads(
                response
            )

            log(
                "CHAT | JSON validation successful"
            )

        except json.JSONDecodeError as error:

            log(
                f"CHAT | JSON validation FAILED | "
                f"line={error.lineno} "
                f"column={error.colno}"
            )

            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "message":
                            "The model did not produce valid JSON.",

                        "type":
                            "invalid_model_output",

                        "code":
                            "invalid_json"
                    }
                }
            )

    else:

        log(
            "CHAT | JSON validation skipped"
        )

    # -----------------------------------------------------
    # Build OpenAI response
    # -----------------------------------------------------

    completion_id = (
        f"chatcmpl-"
        f"{uuid.uuid4().hex}"
    )

    log(
        f"CHAT | Building OpenAI response | "
        f"completion_id={completion_id}"
    )

    result = {
        "id":
            completion_id,

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
                    "role":
                        "assistant",

                    "content":
                        response
                },

                "finish_reason":
                    "stop"
            }
        ],

        "usage": {
            "prompt_tokens":
                prompt_tokens,

            "completion_tokens":
                completion_tokens,

            "total_tokens":
                (
                    prompt_tokens
                    + completion_tokens
                )
        }
    }

    total_chat_time = (
        time.perf_counter()
        - chat_started
    )

    log(
        f"CHAT | Response ready | "
        f"completion_id={completion_id} "
        f"duration={total_chat_time:.2f}s"
    )

    return result


# =========================================================
# Existing Procurement Analyze Endpoint
#
# Preserved for backward compatibility.
#
# POST /v1/procurement/analyze
#
# Supports:
#
# Authorization: Bearer <key>
# X-API-Key: <key>
#
# =========================================================

@app.post("/v1/procurement/analyze")
def analyze(
    request: AnalyzeRequest,

    authorization: str = Header(
        default=""
    ),

    x_api_key: str = Header(
        default=""
    )
):

    analyze_started = (
        time.perf_counter()
    )

    log(
        f"ANALYZE | Request parsed | "
        f"message_chars={len(request.message)}"
    )

    # -----------------------------------------------------
    # Authentication
    # -----------------------------------------------------

    log(
        "ANALYZE | Starting authentication"
    )

    verify_api_key(
        authorization=authorization,
        x_api_key=x_api_key
    )

    log(
        "ANALYZE | Authentication passed"
    )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    log(
        "ANALYZE | Building procurement prompt"
    )

    prompt = (
        "You are a procurement analysis system.\n"
        "Return only valid JSON.\n\n"
        f"USER:\n{request.message}\n\n"
        "ASSISTANT:\n"
    )

    log(
        f"ANALYZE | Prompt ready | "
        f"characters={len(prompt)}"
    )

    # -----------------------------------------------------
    # Generation
    # -----------------------------------------------------

    log(
        "ANALYZE | Sending request to model"
    )

    (
        response,
        prompt_tokens,
        completion_tokens
    ) = generate_response(
        prompt=prompt,
        max_new_tokens=300
    )

    log(
        f"ANALYZE | Model response received | "
        f"prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens}"
    )

    # -----------------------------------------------------
    # JSON validation
    # -----------------------------------------------------

    log(
        "ANALYZE | JSON validation started"
    )

    try:

        result = json.loads(
            response
        )

        log(
            "ANALYZE | JSON validation successful"
        )

    except json.JSONDecodeError as error:

        log(
            f"ANALYZE | JSON validation FAILED | "
            f"line={error.lineno} "
            f"column={error.colno}"
        )

        raise HTTPException(
            status_code=422,
            detail={
                "code":
                    "INVALID_MODEL_OUTPUT",

                "message":
                    "The model did not produce valid JSON."
            }
        )

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    total_analyze_time = (
        time.perf_counter()
        - analyze_started
    )

    log(
        f"ANALYZE | Response ready | "
        f"duration={total_analyze_time:.2f}s"
    )

    return {
        "success":
            True,

        "model":
            "procurement-0.6b-v0.2",

        "analysis":
            result
    }