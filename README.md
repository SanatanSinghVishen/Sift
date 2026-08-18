<div align="center">

# Sift-1B

**Extract the signal. Route the action.**

*A 1.5B-parameter Small Language Model (SLM) fine-tuned via QLoRA + DPO for zero-fluff, deterministic JSON function calling & multi-agent routing.*

[![Documentation](https://img.shields.io/badge/🌐%20Live%20Docs-sift--1--b.vercel.app-black?style=for-the-badge&logo=vercel)](https://sift-1-b.vercel.app/)
[![GitHub Stars](https://img.shields.io/github/stars/SanatanSinghVishen/Sift-1B?style=for-the-badge&logo=github&color=gold)](https://github.com/SanatanSinghVishen/Sift-1B/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Hugging Face GGUF](https://img.shields.io/badge/🤗%20GGUF-sift--1b--gguf-yellow?style=for-the-badge)](https://huggingface.co/SanatanSinghVishen/sift-1b-gguf)
[![Hugging Face DPO](https://img.shields.io/badge/🤗%20DPO%20Adapter-sift--1b--dpo-orange?style=for-the-badge)](https://huggingface.co/SanatanSinghVishen/sift-1b-dpo)

---

### 🌟 If you find Sift-1B useful for your local AI workflows, please consider giving this repository a ⭐ Star!

[Live Documentation](https://sift-1-b.vercel.app/) • [Model Weights](https://huggingface.co/SanatanSinghVishen/sift-1b-gguf) • [Quick Start](#-quick-start) • [Benchmarks](#-benchmarks--evaluation) • [Architecture](#%EF%B8%8F-architecture--pipeline-topology)

</div>

---

## 📌 Table of Contents

- [Overview & Core Mission](#-overview--core-mission)
- [Key Features](#-key-features)
- [Live Interactive Documentation](#-live-interactive-documentation)
- [Benchmarks & Evaluation](#-benchmarks--evaluation)
- [Quick Start](#-quick-start)
  - [1. Ollama (Recommended)](#1-ollama-recommended)
  - [2. Python (OpenAI SDK Compatible)](#2-python-openai-sdk-compatible)
  - [3. Docker Container](#3-docker-container)
  - [4. llama.cpp CLI](#4-llamacpp-cli)
- [Architecture & Pipeline Topology](#%EF%B8%8F-architecture--pipeline-topology)
- [Technical Specifications & Training](#-technical-specifications--training)
- [Project Directory Structure](#-project-directory-structure)
- [Contributing & Community](#-contributing--community)
- [Author & License](#-author--license)

---

## ⚡ Overview & Core Mission

Modern AI applications often rely on massive cloud Large Language Models (LLMs) simply to perform basic intent routing or parameter extraction. This approach introduces high per-token costs, unwanted latency, and potential data privacy risks.

**Sift-1B** is a specialized, local-first Small Language Model (SLM) engineered to:
1. **Sift** structured JSON parameters out of messy, unstructured user prompts.
2. **Shift** the routing workload away from expensive cloud APIs to local edge hardware.

Standard general-purpose LLMs tend to wrap output in conversational preamble (e.g., *"Sure, here is your requested JSON..."*). Sift-1B is explicitly trained via a 2-stage **Supervised Fine-Tuning (SFT) + Direct Preference Optimization (DPO)** pipeline to eliminate all markdown noise and output strict, machine-readable JSON on the first attempt.

---

## 🎯 Key Features

- **100% Tool Selection Accuracy:** Perfectly identifies the target function schema from holdout test cases.
- **Zero Markdown Fluff:** Eliminates preamble conversational text (`"Here is your result"`) and markdown blocks (` ```json `) for instant parsing.
- **Sub-35ms First Token Latency:** Runs natively on local hardware (4 GB VRAM RTX 3050, Apple Silicon M-series, or edge CPUs).
- **OpenAI API Compatibility:** Drop-in replacement for OpenAI SDK clients (`base_url="http://localhost:8000/v1"`).
- **Total Data Sovereignty:** Operates 100% offline with zero external API calls or telemetry.

---

## 🌐 Live Interactive Documentation

Explore the complete interactive documentation, architecture deep dives, and live integration snippets at:

👉 **[https://sift-1-b.vercel.app/](https://sift-1-b.vercel.app/)**

---

## 📊 Benchmarks & Evaluation

Evaluated across **50 holdout test cases containing UNSEEN function schemas** (not present in the training set) comparing Base Qwen vs. SFT vs. DPO (Golden Release Checkpoint-750):

| Metric | Qwen2.5-1.5B (Base) | Sift-1B (SFT) | 🏆 **Sift-1B (DPO Aligned - Ours)** | Delta vs Base |
| :--- | :---: | :---: | :---: | :---: |
| **Tool Selection Accuracy** | 70.0% | 98.0% | **100.0%** 🏆 | **+30.0%** |
| **Parameter Extraction Accuracy** | 34.0% | 80.0% | **88.0%** 🏆 | **+54.0%** |
| **JSON Parse Rate** | 96.0% | 98.0% | **100.0%** 🏆 | **+4.0%** |
| **Zero Markdown / Fluff Rate** | 76.0% | 100.0% | **100.0%** 🏆 | **+24.0%** |
| **Zero Hallucination Rate** | 100.0% | 100.0% | **100.0%** 🏆 | **0% Hallucinations** |
| **Average Latency (TTFT)** | 2,277 ms | 1,734 ms | **1,714 ms** ⚡ | **25% Faster** |

---

## 🚀 Quick Start

### 1. Ollama (Recommended)

Run Sift-1B locally with a single command:

```bash
ollama run hf.co/SanatanSinghVishen/sift-1b-gguf
```

Or build locally with custom `Modelfile`:

```dockerfile
FROM ./export/sift-1b/sift-1b-q4_k_m.gguf
PARAMETER temperature 0.1
PARAMETER top_p 0.95
SYSTEM "You are a function calling agent. Output only valid JSON tool calls."
```

```bash
ollama create sift-1b -f Modelfile
ollama run sift-1b "Schedule a team sync tomorrow at 10 AM"
```

---

### 2. Python (OpenAI SDK Compatible)

Start the local server (`python serve/server.py`) and point your existing OpenAI SDK client to `localhost`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="sift-1b",
    messages=[
        {
            "role": "system", 
            "content": "You are a function calling agent. Output only valid JSON tool calls."
        },
        {
            "role": "user", 
            "content": "Book a ride to JFK airport arriving by 5:00 PM."
        }
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "book_ride",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "arrival_time": {"type": "string"}
                },
                "required": ["destination", "arrival_time"]
            }
        }
    }]
)

print(response.choices[0].message.tool_calls)
```

---

### 3. Docker Container

Deploy the containerized server with GPU acceleration:

```bash
docker run -d -p 8000:8000 --gpus all ghcr.io/sanatansinghvishen/sift-1b:latest
```

---

### 4. llama.cpp CLI

Run low-level GGUF inference directly from the command line:

```bash
./llama-cli -m sift-1b-q4_k_m.gguf \
  --n-gpu-layers -1 \
  --temp 0 \
  -p "<|im_start|>system\nYou are a function calling agent. Output only valid JSON tool calls.<|im_end|>\n<|im_start|>user\nWhat is the weather in NYC?<|im_end|>\n<|im_start|>assistant\n"
```

---

## 🏗️ Architecture & Pipeline Topology

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

---

## ⚙️ Technical Specifications & Training

| Parameter | Value |
| :--- | :--- |
| **Base Architecture** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Quantization** | QLoRA 4-bit NF4 (`load_in_4bit=True`) |
| **LoRA Config** | Rank $r=16$, Alpha $\alpha=32$, Dropout $0$ |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **SFT Dataset** | 10,000 ChatML multi-turn function calling rows (Salesforce/xlam-60k sample) |
| **DPO Dataset** | 10,000 synthetic preference pairs ($y_w$ clean JSON vs $y_l$ markdown fluff) |
| **DPO Temperature ($\beta$)** | `0.1` (Sigmoid loss) |
| **Training Compute** | 1x NVIDIA RTX 3050 GPU (4 GB VRAM budget) |
| **Frameworks** | Unsloth + TRL + Hugging Face PEFT |

---

## 📁 Project Directory Structure

```
Sift-1B/
├── data/           # Dataset preparation & DPO mutation scripts
├── training/       # SFT & DPO training loops (Unsloth + TRL)
├── eval/           # Evaluation metrics & latency benchmark suite
├── export/         # GGUF conversion & Hugging Face push utilities
├── serve/          # FastAPI OpenAI-compatible inference server
└── web/            # Vercel documentation platform (Next.js 16 + Lenis)
```

---

## ⭐ Contributing & Community

Contributions, issues, and feature requests are welcome!  
If you find this project helpful, please support it by giving it a **⭐ Star** on GitHub!

1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author & License

Designed and developed by **Sanatan Singh**.

- **Email:** [sanatansinghgonda@gmail.com](mailto:sanatansinghgonda@gmail.com)
- **GitHub:** [@SanatanSinghVishen](https://github.com/SanatanSinghVishen)
- **LinkedIn:** [Sanatan Singh](https://www.linkedin.com/in/sanatansingh380/)
- **Hugging Face:** [@SanatanSinghVishen](https://huggingface.co/SanatanSinghVishen)

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
