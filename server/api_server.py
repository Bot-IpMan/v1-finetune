"""
Simple OpenAI‑compatible API server for serving a fine‑tuned model.

This server loads a base model and optionally a LoRA adapter from disk and exposes
a `/v1/chat/completions` endpoint compatible with OpenAI's Chat API.  It is
intended to be used with Open WebUI which allows connecting to arbitrary
OpenAI‑compatible backends【362441291072753†L54-L131】.

Environment variables:

* ``MODEL_PATH`` – Path to the directory containing the fine‑tuned model.  If a
  LoRA adapter was used the directory should contain both the adapter weights
  and a file called ``base_model_name.txt`` with the name of the base model.
* ``BASE_MODEL_NAME`` – (optional) Explicitly provide the base model name.  This
  can be used when no ``base_model_name.txt`` file is present.

Example:

    MODEL_PATH=/models/finetuned BASE_MODEL_NAME=Qwen/Qwen1.5-7B uvicorn api_server:app
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI()

# Read configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/sdb/data/ai/models/model_output")
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", None)


def load_model_and_tokenizer():
    """Load the tokenizer and model with optional LoRA."""
    global tokenizer, model
    # Determine the base model name.  If the LoRA adapter directory contains a
    # marker file write by train.py use that, otherwise fall back to the
    # BASE_MODEL_NAME environment variable.
    base_name: Optional[str] = BASE_MODEL_NAME
    base_name_file = os.path.join(MODEL_PATH, "base_model_name.txt")
    if os.path.exists(base_name_file):
        with open(base_name_file, "r", encoding="utf-8") as f:
            base_name = f.read().strip()

    try:
        # Load tokenizer.  The LoRA adapter directory may not contain a tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            base_name or MODEL_PATH, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            base_name or MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        # If the model directory contains LoRA weights, load them
        adapter_path = os.path.join(MODEL_PATH, "adapter_model.bin")
        if os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(model, MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as exc:
        # Fallback to a dummy tokenizer and no model.  This allows the server to
        # start even if the model has not been trained yet.
        print(f"Warning: failed to load model from {MODEL_PATH}: {exc}")
        fallback_name = base_name or "gpt2"
        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            tokenizer = None
        model = None
        return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


class Choice(BaseModel):
    message: ChatMessage
    index: int
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str
    choices: List[Choice]
    usage: Dict[str, int] = {}


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest):
    """Generate a chat completion from the model."""
    # Convert incoming pydantic models to simple dict/list expected by tokenizer
    msgs = [m.dict() for m in req.messages]
    # Build prompt using the chat template
    # If model is not loaded return a fallback message
    if model is None or tokenizer is None:
        content = "Model is not loaded. Please train the model before querying."
        response_message = ChatMessage(role="assistant", content=content)
        return ChatResponse(
            id="cmpl-0",
            object="chat.completion",
            choices=[Choice(message=response_message, index=0, finish_reason="stop")],
            usage={},
        )

    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate completion
    generation_kwargs = {
        "max_new_tokens": req.max_tokens or 512,
        "temperature": req.temperature or 0.7,
        "do_sample": True,
        "top_p": 0.95,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    # Extract only the generated part
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response_message = ChatMessage(role="assistant", content=generated_text)
    return ChatResponse(
        id="cmpl-1",
        object="chat.completion",
        choices=[Choice(message=response_message, index=0, finish_reason="stop")],
        usage={},
    )
