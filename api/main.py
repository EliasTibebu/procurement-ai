import json
import os
import threading

import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "models/qwen3-0.6b-base"
ADAPTER = "output/procurement-0.6b-lora-v0.2"

API_KEY = os.getenv(
    "PROCUREMENT_AI_API_KEY",
    "change-me"
)

torch.set_num_threads(6)
torch.set_num_interop_threads(1)


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="Procurement AI",
    version="0.2.0"
)


# ---------------------------------------------------------
# Request
# ---------------------------------------------------------

class AnalyzeRequest(BaseModel):
    message: str


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

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


# CPU generation should not run concurrently
generation_lock = threading.Lock()


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "qwen3-0.6b",
        "adapter": "procurement-v0.2"
    }


# ---------------------------------------------------------
# Procurement analysis
# ---------------------------------------------------------

@app.post("/v1/procurement/analyze")
def analyze(
    request: AnalyzeRequest,
    x_api_key: str = Header(default="")
):

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    prompt = (
        "You are a procurement analysis system.\n"
        "Return only valid JSON.\n\n"
        f"USER:\n{request.message}\n\n"
        "ASSISTANT:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    try:

        with generation_lock:

            with torch.inference_mode():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.eos_token_id
                )

        generated = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        result = json.loads(response)

        return {
            "success": True,
            "model": "procurement-0.6b-v0.2",
            "analysis": result
        }

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_MODEL_OUTPUT",
                "message":
                    "The model did not produce valid JSON."
            }
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
