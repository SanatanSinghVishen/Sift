"""
Sift — Full-Fledged 3-Way Comparative Benchmark
=================================================
Compares:
  1. Base Model:  Qwen/Qwen2.5-1.5B-Instruct (Zero-Shot)
  2. SFT Model:   SanatanSinghVishen/sift-1b-sft (Supervised Fine-Tuning)
  3. DPO Model:   SanatanSinghVishen/sift-1b-dpo (Direct Preference Optimization)

Metrics:
  - JSON Parse Rate (%):      Output is strictly valid JSON
  - Clean Format Rate (%):    Zero markdown codeblocks (```json) & zero conversational fluff
  - Tool Selection Acc (%):   Correct function/tool selected
  - Param Extraction Acc (%): Correct parameter keys and values extracted
  - Zero Hallucination (%):   No invented/hallucinated parameters
  - Inference Latency (ms):   Time To First Token / generation speed
"""

import sys
import json
import time
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console(legacy_windows=False)


def normalize_tool_calls(obj):
    """
    Normalizes function calls from various JSON formats:
    - [{"name": "f", "arguments": {...}}]
    - [["f", {...}]]
    - {"name": "f", "parameters": {...}}
    Returns a normalized list: [{"name": str, "arguments": dict}]
    """
    normalized = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                name = item.get("name") or item.get("function") or ""
                args = item.get("arguments") or item.get("parameters") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                normalized.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
            elif isinstance(item, list) and len(item) >= 2:
                name = item[0]
                args = item[1] if isinstance(item[1], dict) else {}
                normalized.append({"name": str(name), "arguments": args})
    elif isinstance(obj, dict):
        name = obj.get("name") or obj.get("function") or ""
        args = obj.get("arguments") or obj.get("parameters") or {}
        normalized.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
    return normalized


def evaluate_response(generated_text: str, expected_text: str, schema_keys: set) -> dict:
    """Evaluates a single model generation against expected ground truth."""
    metrics = {
        "valid_json": False,
        "clean_format": False,
        "tool_match": False,
        "param_match": False,
        "no_hallucination": True,
    }

    # 1. Check Format Cleanliness (Zero Markdown wrappers, Zero conversational filler)
    format_errors = [
        "```json", "```", "sure", "certainly", "here is",
        "let me", "of course", "i'd be happy", "great question", "i can help",
    ]
    lower = generated_text.lower()
    metrics["clean_format"] = not any(err in lower for err in format_errors)

    # 2. Check JSON validity
    try:
        # Strip potential markdown if testing base model
        clean_text = generated_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[-1].split("```")[0].strip()

        parsed_gen = json.loads(clean_text)
        metrics["valid_json"] = True
    except Exception:
        return metrics

    try:
        parsed_exp = json.loads(expected_text)
    except Exception:
        parsed_exp = []

    norm_gen = normalize_tool_calls(parsed_gen)
    norm_exp = normalize_tool_calls(parsed_exp)

    # 3. Tool Selection Match
    gen_tools = [t["name"] for t in norm_gen]
    exp_tools = [t["name"] for t in norm_exp]
    metrics["tool_match"] = (gen_tools == exp_tools) and len(gen_tools) > 0

    # 4. Parameter Extraction Match
    if metrics["tool_match"]:
        all_params_match = True
        for g_tool, e_tool in zip(norm_gen, norm_exp):
            g_args = g_tool["arguments"]
            e_args = e_tool["arguments"]
            # Check subset/equality of expected parameters
            for k, v in e_args.items():
                if k not in g_args or str(g_args[k]).lower() != str(v).lower():
                    all_params_match = False
                    break
        metrics["param_match"] = all_params_match

    # 5. Hallucination Check
    if schema_keys:
        for g_tool in norm_gen:
            for k in g_tool["arguments"].keys():
                if k not in schema_keys and k not in ("name", "arguments", "parameters"):
                    metrics["no_hallucination"] = False
                    break

    return metrics


def extract_schema_keys(system_content: str) -> set:
    keys = set()
    try:
        start = system_content.find("[")
        if start == -1:
            start = system_content.find("{")
        if start != -1:
            tools = json.loads(system_content[start:])
            if isinstance(tools, list):
                for t in tools:
                    params = t.get("function", {}).get("parameters", {}).get("properties", {}) or t.get("parameters", {}).get("properties", {})
                    if isinstance(params, dict):
                        keys.update(params.keys())
    except Exception:
        pass
    return keys


def run_benchmark_for_model(model, tokenizer, test_samples: list, model_name: str) -> dict:
    results = []
    total_time = 0.0

    for sample in track(test_samples, description=f"Evaluating {model_name}..."):
        convs = sample["conversations"]
        prompt_msgs = [c for c in convs if c["role"] in ("system", "user")]
        expected = [c["content"] for c in convs if c["role"] == "assistant"][0]
        schema_keys = extract_schema_keys(prompt_msgs[0]["content"])

        encoded = tokenizer.apply_chat_template(prompt_msgs, return_tensors="pt", add_generation_prompt=True)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) or hasattr(encoded, "input_ids") else encoded
        input_ids = input_ids.to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_time += elapsed_ms

        generated = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        eval_res = evaluate_response(generated, expected, schema_keys)
        eval_res["latency_ms"] = elapsed_ms
        results.append(eval_res)

    N = len(results)
    return {
        "model": model_name,
        "samples": N,
        "json_parse_rate": sum(1 for r in results if r["valid_json"]) / N * 100,
        "clean_format_rate": sum(1 for r in results if r["clean_format"]) / N * 100,
        "tool_selection_acc": sum(1 for r in results if r["tool_match"]) / N * 100,
        "param_extraction_acc": sum(1 for r in results if r["param_match"]) / N * 100,
        "zero_hallucination_rate": sum(1 for r in results if r["no_hallucination"]) / N * 100,
        "avg_latency_ms": total_time / N,
    }


def main():
    parser = argparse.ArgumentParser(description="Full-Fledged Sift 3-Way Benchmark")
    parser.add_argument("--samples", type=int, default=20, help="Number of holdout test samples")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]================================================================[/bold cyan]")
    console.print(f"[bold cyan]       SIFT-1B COMPREHENSIVE 3-WAY MODEL BENCHMARK             [/bold cyan]")
    console.print(f"[bold cyan]================================================================[/bold cyan]")
    console.print(f"  Holdout Test Samples: {args.samples}")
    console.print(f"  Tiers Evaluated:      Base Qwen 1.5B vs. SFT vs. DPO (Ours)\n")

    # Load holdout dataset
    rows = []
    with open("data/sft_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    test_samples = rows[-args.samples:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_id = "Qwen/Qwen2.5-1.5B-Instruct"

    console.print(f"[yellow]⏳ Loading Base Model: {base_id} on {device}...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        base_model = base_model.to("cpu")
    base_model.eval()

    # 1. Evaluate Base Model
    console.print("\n[bold]Phase 1/3: Benchmarking Base Model (Zero-Shot)[/bold]")
    base_metrics = run_benchmark_for_model(base_model, tokenizer, test_samples, "Qwen2.5-1.5B (Base)")

    # 2. Evaluate SFT Model
    console.print("\n[bold]Phase 2/3: Benchmarking SFT Model[/bold]")
    sft_model = PeftModel.from_pretrained(base_model, "SanatanSinghVishen/sift-1b-sft")
    sft_model.eval()
    sft_metrics = run_benchmark_for_model(sft_model, tokenizer, test_samples, "Sift-1B (SFT)")
    del sft_model

    # 3. Evaluate DPO Model
    console.print("\n[bold]Phase 3/3: Benchmarking DPO Model (Final Alignment)[/bold]")
    dpo_model = PeftModel.from_pretrained(base_model, "SanatanSinghVishen/sift-1b-dpo")
    dpo_model.eval()
    dpo_metrics = run_benchmark_for_model(dpo_model, tokenizer, test_samples, "Sift-1B (DPO Aligned)")

    # =========================================================================
    # Print Publication Comparison Table
    # =========================================================================
    table = Table(title="Sift-1B Function-Calling Benchmark Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Benchmark Metric", style="cyan", width=26)
    table.add_column("Qwen-1.5B (Base)", justify="right", width=18)
    table.add_column("Sift-1B (SFT)", justify="right", width=18)
    table.add_column("Sift-1B (DPO - Ours)", justify="right", style="bold green", width=22)

    table.add_row(
        "JSON Parse Rate",
        f"{base_metrics['json_parse_rate']:.1f}%",
        f"{sft_metrics['json_parse_rate']:.1f}%",
        f"{dpo_metrics['json_parse_rate']:.1f}%",
    )
    table.add_row(
        "Zero Markdown / Fluff",
        f"{base_metrics['clean_format_rate']:.1f}%",
        f"{sft_metrics['clean_format_rate']:.1f}%",
        f"{dpo_metrics['clean_format_rate']:.1f}%",
    )
    table.add_row(
        "Tool Selection Acc",
        f"{base_metrics['tool_selection_acc']:.1f}%",
        f"{sft_metrics['tool_selection_acc']:.1f}%",
        f"{dpo_metrics['tool_selection_acc']:.1f}%",
    )
    table.add_row(
        "Param Extraction Acc",
        f"{base_metrics['param_extraction_acc']:.1f}%",
        f"{sft_metrics['param_extraction_acc']:.1f}%",
        f"{dpo_metrics['param_extraction_acc']:.1f}%",
    )
    table.add_row(
        "Zero Hallucination Rate",
        f"{base_metrics['zero_hallucination_rate']:.1f}%",
        f"{sft_metrics['zero_hallucination_rate']:.1f}%",
        f"{dpo_metrics['zero_hallucination_rate']:.1f}%",
    )
    table.add_row(
        "Avg Latency",
        f"{base_metrics['avg_latency_ms']:.0f} ms",
        f"{sft_metrics['avg_latency_ms']:.0f} ms",
        f"{dpo_metrics['avg_latency_ms']:.0f} ms",
    )

    console.print("\n")
    console.print(table)

    # Save to json and markdown
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": args.samples,
        "models": {
            "base": base_metrics,
            "sft": sft_metrics,
            "dpo": dpo_metrics,
        }
    }
    with open("eval/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    console.print("\n[bold green]✓ Complete Benchmark report saved to eval/benchmark_results.json![/bold green]\n")


if __name__ == "__main__":
    main()
