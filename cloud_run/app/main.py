import json
import logging
import os
import time
import uuid
from functools import lru_cache

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gemmalaw")

API_KEY = os.getenv("API_KEY", "dev-key")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "google/gemma-3-4b-it")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH")
LORA_PATH = os.getenv("LORA_PATH", "/app/models/lora")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

app = FastAPI(title="GemmaLaw Inference Server")


class LegalAnswerRequest(BaseModel):
    query: str = Field(..., description="User question or prompt")
    max_new_tokens: int = Field(512, ge=1, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    compare_base: bool = Field(False, description="Return both base and LoRA outputs")


class LegalAnswerResponse(BaseModel):
    request_id: str
    input_tokens: int
    lora_output_tokens: int
    lora_text: str
    base_output_tokens: int | None = None
    base_text: str | None = None
    latency_ms: float


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        logger.warning(
            json.dumps({
                "event": "auth_failed",
                "provided": x_api_key,
                "expected": "set" if API_KEY else "unset",
            })
        )
        raise HTTPException(status_code=401, detail="unauthorized")


@lru_cache(maxsize=2)
def load_tokenizer():
    model_id = BASE_MODEL_PATH if BASE_MODEL_PATH else BASE_MODEL_ID
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


@lru_cache(maxsize=1)
def load_model():
    model_id = BASE_MODEL_PATH if BASE_MODEL_PATH else BASE_MODEL_ID
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=DTYPE,
    )
    peft_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    peft_model.to(DEVICE)
    peft_model.eval()
    return peft_model


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/legal/answer", response_model=LegalAnswerResponse)
def legal_answer(payload: LegalAnswerRequest, _: None = Depends(require_api_key)):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    tokenizer = load_tokenizer()
    model = load_model()

    inputs = tokenizer(payload.query, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    gen_kwargs = {
        "max_new_tokens": payload.max_new_tokens,
        "temperature": payload.temperature,
        "top_p": payload.top_p,
        "do_sample": payload.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    output_ids = generated[0, input_len:]
    lora_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    base_text = None
    base_output_tokens = None

    if payload.compare_base:
        active_adapters = getattr(model, "active_adapters", None)

        with torch.inference_mode():
            model.disable_adapter()
            base_generated = model.base_model.generate(**inputs, **gen_kwargs)
            if active_adapters:
                model.set_adapter(active_adapters)
        base_output_ids = base_generated[0, input_len:]
        base_text = tokenizer.decode(base_output_ids, skip_special_tokens=True)
        base_output_tokens = base_output_ids.shape[0]

    latency = (time.perf_counter() - started) * 1000
    response = LegalAnswerResponse(
        request_id=request_id,
        input_tokens=inputs["input_ids"].shape[1],
        lora_output_tokens=output_ids.shape[0],
        lora_text=lora_text.strip(),
        base_output_tokens=base_output_tokens,
        base_text=base_text.strip() if base_text else None,
        latency_ms=latency,
    )

    logger.info(json.dumps({
        "event": "inference",
        "request_id": request_id,
        "input_tokens": response.input_tokens,
        "lora_output_tokens": response.lora_output_tokens,
        "compare_base": payload.compare_base,
        "latency_ms": latency,
    }))

    return JSONResponse(status_code=200, content=json.loads(response.json()))
