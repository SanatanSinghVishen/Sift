<div align="center">

# Sift-1B

**Extract the signal. Route the action.**

A 1.5B-parameter Small Language Model fine-tuned via QLoRA + DPO for deterministic JSON function calling.  
Runs entirely on your local hardware.

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-SanatanSinghVishen%2Fsift--1b-yellow)](https://huggingface.co/SanatanSinghVishen/sift-1b)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## What is Sift?

Sift is a **hyper-specialized** small language model that does one thing perfectly: it **sifts** structured JSON parameters out of unstructured user prompts and **shifts** the routing workload away from expensive cloud LLMs.

Standard LLMs wrap their outputs in conversational fluff. Sift has been explicitly trained — through a two-stage SFT + DPO pipeline — to strip away all noise and output strict, type-safe, machine-readable JSON on the first attempt.

## Quick Start

### Option 1: Ollama (Recommended)
```bash
ollama run SanatanSinghVishen/sift-1b
```

### Option 2: Python (OpenAI-Compatible)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="sift-1b",
    messages=[
        {"role": "system", "content": "You are a function calling agent. Output only valid JSON tool calls."},
        {"role": "user", "content": "Schedule a team sync for tomorrow at 10 AM"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "create_event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "time": {"type": "string"}
                },
                "required": ["title", "time"]
            }
        }
    }]
)
print(response.choices[0].message.tool_calls)
```

### Option 3: Docker
```bash
docker run -d -p 8000:8000 --gpus all ghcr.io/sanatansinghvishen/sift-1b:latest
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                      │
│                                                             │
│  [Salesforce/xlam-60k]                                      │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐   QLoRA (4-bit)   ┌──────────────────┐   │
│  │ Qwen2.5-1.5B │ ───────────────►  │ SFT Checkpoint   │   │
│  └──────────────┘                   └──────────────────┘   │
│                                            │                │
│                                            ▼                │
│  [Mutated Preference Pairs]  ───►  DPO Alignment            │
│                                            │                │
│                                            ▼                │
│                                   Merged GGUF Weights       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION RUNTIME                        │
│                                                             │
│  Incoming Request ──► [FastAPI / llama-cpp] ──► JSON Output │
│                         (Sub-35ms TTFT)                     │
└─────────────────────────────────────────────────────────────┘
```

## Training Details

| Component | Value |
|---|---|
| Base Model | Qwen2.5-1.5B-Instruct |
| Quantization | QLoRA (4-bit NF4) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| SFT Dataset | Salesforce/xlam-function-calling-60k (10k sample) |
| DPO Dataset | Synthetically mutated preference pairs |
| Training Hardware | NVIDIA RTX 3050 (4 GB VRAM) |
| Framework | Unsloth + TRL |

## Benchmarks

| Metric | Sift-1B | Base Qwen2.5-1.5B |
|---|---|---|
| Schema Adherence | TBD | TBD |
| Hallucination Rate | TBD | TBD |
| Format Error Rate | TBD | TBD |
| TTFT (GGUF Q4_K_M) | TBD | — |

## Project Structure

```
Sift/
├── data/           # Dataset preparation & DPO mutation scripts
├── training/       # SFT & DPO training loops (Unsloth + TRL)
├── eval/           # Evaluation metrics & latency benchmarks
├── export/         # GGUF conversion & Hugging Face push
├── serve/          # FastAPI inference server
└── web/            # Vercel documentation hub (Next.js)
```

## License

MIT
