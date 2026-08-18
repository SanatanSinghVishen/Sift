"""
Sift — FastAPI Inference Server
=================================
A lightweight, OpenAI-compatible REST API that serves the Sift-1B
GGUF model using llama-cpp-python.

Developers can point their existing OpenAI SDKs to this server
by simply changing the base_url to http://localhost:8000/v1.

Usage:
  python serve/server.py
  python serve/server.py --model export/sift-1b/unsloth.Q4_K_M.gguf --port 8000

API Endpoints:
  POST /v1/chat/completions    — OpenAI-compatible chat completions
  GET  /v1/models              — List available models
  GET  /health                 — Health check
"""

import os
import json
import time
import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="Sift-1B Inference Server",
    description="OpenAI-compatible API for the Sift-1B function-calling SLM",
    version="1.0.0",
)

# CORS — allow the Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference (initialized on startup)
llm = None
MODEL_PATH = None


# =============================================================================
# Request/Response Models (OpenAI-Compatible)
# =============================================================================

class Message(BaseModel):
    role: str
    content: str


class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: dict = {}


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str = "sift-1b"
    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "model_loaded": llm is not None,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "sift-1b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "SanatanSinghVishen",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    
    If tools are provided, they are injected into the system prompt
    so the model knows which functions are available.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build the system prompt with tool definitions
    messages = []
    tools_injected = False

    for msg in request.messages:
        if msg.role == "system" and request.tools and not tools_injected:
            # Append tool definitions to the system message
            tools_json = json.dumps(
                [{"name": t.function.name, "description": t.function.description,
                  "parameters": t.function.parameters} for t in request.tools],
                indent=2,
            )
            enhanced_content = (
                f"{msg.content}\n\n"
                f"Available tools:\n{tools_json}"
            )
            messages.append({"role": "system", "content": enhanced_content})
            tools_injected = True
        else:
            messages.append({"role": msg.role, "content": msg.content})

    # If no system message was present but tools were provided, add one
    if request.tools and not tools_injected:
        tools_json = json.dumps(
            [{"name": t.function.name, "description": t.function.description,
              "parameters": t.function.parameters} for t in request.tools],
            indent=2,
        )
        system_msg = {
            "role": "system",
            "content": (
                "You are a strict function calling agent. Output ONLY valid JSON "
                "function calls. No explanation, no markdown.\n\n"
                f"Available tools:\n{tools_json}"
            ),
        }
        messages.insert(0, system_msg)

    # Generate response
    start_time = time.perf_counter()

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Extract generated text
    generated = response["choices"][0]["message"]["content"]

    # Build OpenAI-compatible response
    return ChatCompletionResponse(
        id=f"chatcmpl-sift-{int(time.time())}",
        created=int(time.time()),
        model="sift-1b",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=generated),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        ),
    )


# =============================================================================
# Inference Engine (Dual Support: GGUF via llama-cpp & HuggingFace via PEFT)
# =============================================================================

class HuggingFaceEngine:
    """Fallback PyTorch / HuggingFace inference engine for serving adapters directly."""
    def __init__(self, model_id_or_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_id = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"⏳ Loading base model {base_id} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(base_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            base_model = base_model.to("cpu")

        print(f"⏳ Attaching adapter from {model_id_or_path}...")
        self.model = PeftModel.from_pretrained(base_model, model_id_or_path)
        self.model.eval()
        print(f"✓ Sift DPO model ready for inference on {self.device}!")

    def create_chat_completion(self, messages, max_tokens=256, temperature=0.0, top_p=1.0):
        prompt_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
            )
        gen_ids = outputs[0][prompt_inputs.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": prompt_inputs.shape[-1],
                "completion_tokens": len(gen_ids),
                "total_tokens": prompt_inputs.shape[-1] + len(gen_ids),
            }
        }


def load_model(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048):
    """Load model using GGUF (llama-cpp) or Hugging Face adapter."""
    global llm, MODEL_PATH
    MODEL_PATH = model_path

    if model_path.endswith(".gguf") and Path(model_path).exists():
        from llama_cpp import Llama
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        print(f"✓ GGUF Model loaded: {model_path}")
    else:
        llm = HuggingFaceEngine(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sift-1B Inference Server")
    parser.add_argument(
        "--model", type=str,
        default="SanatanSinghVishen/sift-1b-dpo",
        help="Path to GGUF model file or HuggingFace adapter ID",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                       help="GPU layers to offload (-1 = all)")
    parser.add_argument("--n-ctx", type=int, default=2048,
                       help="Context window size")
    args = parser.parse_args()

    load_model(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx)
    uvicorn.run(app, host=args.host, port=args.port)
