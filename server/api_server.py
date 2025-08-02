"""
Simple OpenAI‑compatible API server for serving a fine‑tuned model.

This server loads a base model and optionally a LoRA adapter from disk and exposes
a `/v1/chat/completions` endpoint compatible with OpenAI's Chat API.  It is
intended to be used with Open WebUI which allows connecting to arbitrary
OpenAI‑compatible backends.

Environment variables:

* ``MODEL_PATH`` – Path to the directory containing the fine‑tuned model.  If a
  LoRA adapter was used the directory should contain both the adapter weights
  and a file called ``base_model_name.txt`` with the name of the base model.
* ``BASE_MODEL_NAME`` – (optional) Explicitly provide the base model name.  This
  can be used when no ``base_model_name.txt`` file is present.

Example:

    MODEL_PATH=/models/finetuned BASE_MODEL_NAME=Qwen/Qwen2.5-Coder-7B uvicorn api_server:app
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="Fine-tuned Model API", version="1.0.0")

# Read configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model_output")
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", None)

# Global variables for model and tokenizer
tokenizer = None
model = None


def load_model_and_tokenizer():
    """Load the tokenizer and model with optional LoRA."""
    global tokenizer, model
    
    # Determine the base model name.  If the LoRA adapter directory contains a
    # marker file written by train.py use that, otherwise fall back to the
    # BASE_MODEL_NAME environment variable.
    base_name: Optional[str] = BASE_MODEL_NAME
    base_name_file = os.path.join(MODEL_PATH, "base_model_name.txt")
    
    if os.path.exists(base_name_file):
        with open(base_name_file, "r", encoding="utf-8") as f:
            base_name = f.read().strip()
            print(f"Using base model from file: {base_name}")

    if not base_name:
        raise ValueError("No base model name provided via BASE_MODEL_NAME env var or base_model_name.txt")

    try:
        print(f"Loading tokenizer from {base_name}")
        # Load tokenizer.  The LoRA adapter directory may not contain a tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            base_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading base model from {base_name}")
        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        # If the model directory contains LoRA weights, load them
        adapter_config_path = os.path.join(MODEL_PATH, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print(f"Loading LoRA adapter from {MODEL_PATH}")
            model = PeftModel.from_pretrained(model, MODEL_PATH)
            print("LoRA adapter loaded successfully")
        else:
            print("No LoRA adapter found, using base model only")
            
        model.eval()
        print("Model loaded and ready for inference")
        return tokenizer, model
        
    except Exception as exc:
        print(f"Error loading model from {MODEL_PATH}: {exc}")
        raise


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


@app.get("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "qwen-finetuned",  # id моделі, має співпадати з DEFAULT_MODELS у docker-compose
                "object": "model",
                "created": 0,
                "owned_by": "user"
            }
        ]
    }

@app.get("/")
def root():
    """Root endpoint with basic info."""
    return {
        "message": "Fine-tuned Model API",
        "model_path": MODEL_PATH,
        "base_model": BASE_MODEL_NAME,
        "model_loaded": model is not None,
        "endpoints": ["/health", "/v1/chat/completions"]
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest):
    """Generate a chat completion from the model."""
    # Check if model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Please wait for model initialization."
        )

    try:
        # Convert incoming pydantic models to simple dict/list expected by tokenizer
        msgs = [m.dict() for m in req.messages]
        
        # Build prompt using the chat template
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
            "pad_token_id": tokenizer.pad_token_id,
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
            usage={"prompt_tokens": len(inputs["input_ids"][0]), "completion_tokens": len(generated_ids)},
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when the application starts."""
    global tokenizer, model
    try:
        print("Initializing model...")
        tokenizer, model = load_model_and_tokenizer()
        print("Model initialization completed successfully")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        # Don't fail startup, let health check handle it
        tokenizer, model = None, None