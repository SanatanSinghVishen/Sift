"""
Sift — Automated Checkpoint Sweep & Leaderboard
=================================================
Scans all checkpoints in Google Drive / local disk, evaluates each
on a holdout test set, ranks them across all metrics, and automatically
picks the optimal #1 checkpoint!

Usage:
  python eval/sweep_checkpoints.py
  python eval/sweep_checkpoints.py --dir /content/drive/MyDrive/sift_dpo_backup --samples 30
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


def evaluate_single(model, tokenizer, conversations, expected_output, schema_keys):
    prompt_messages = [c for c in conversations if c["role"] in ("system", "user")]

    encoded = tokenizer.apply_chat_template(prompt_messages, return_tensors="pt", add_generation_prompt=True)
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

    generated_ids = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    format_errors = ["```json", "```", "sure", "certainly", "here is", "let me", "of course", "i'd be happy", "great question"]
    clean_format = not any(err in generated_text.lower() for err in format_errors)

    valid_json = False
    try:
        clean_text = generated_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[-1].split("```")[0].strip()
        parsed_gen = json.loads(clean_text)
        valid_json = True
    except Exception:
        parsed_gen = []

    try:
        parsed_exp = json.loads(expected_output)
    except Exception:
        parsed_exp = []

    norm_gen = normalize_tool_calls(parsed_gen)
    norm_exp = normalize_tool_calls(parsed_exp)

    gen_tools = [t["name"] for t in norm_gen]
    exp_tools = [t["name"] for t in norm_exp]
    tool_match = (gen_tools == exp_tools) and len(gen_tools) > 0

    param_match = False
    if tool_match:
        all_params = True
        for g, e in zip(norm_gen, norm_exp):
            g_args = g["arguments"]
            e_args = e["arguments"]
            for k, v in e_args.items():
                if k not in g_args or str(g_args[k]).lower() != str(v).lower():
                    all_params = False
                    break
        param_match = all_params

    no_hallucination = True
    if schema_keys and norm_gen:
        for g in norm_gen:
            for k in g["arguments"].keys():
                if k not in schema_keys and k not in ("name", "arguments", "parameters"):
                    no_hallucination = False
                    break

    return {
        "valid_json": valid_json,
        "clean_format": clean_format,
        "tool_match": tool_match,
        "param_match": param_match,
        "no_hallucination": no_hallucination,
        "latency_ms": elapsed_ms,
    }


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


def main():
    parser = argparse.ArgumentParser(description="Sweep all checkpoints and find the best one")
    parser.add_argument(
        "--dir", type=str,
        default="/content/drive/MyDrive/sift_dpo_backup",
        help="Directory containing checkpoint-* folders",
    )
    parser.add_argument("--samples", type=int, default=30, help="Holdout test samples per checkpoint")
    args = parser.parse_args()

    ckpt_dir = Path(args.dir)
    if not ckpt_dir.exists():
        ckpt_dir = Path("checkpoints/dpo")

    ckpts = sorted(
        [d for d in ckpt_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
    )

    if not ckpts:
        console.print(f"[bold red]❌ No checkpoint folders found in {ckpt_dir}![/bold red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]================================================================[/bold cyan]")
    console.print(f"[bold cyan]        SIFT-1B AUTOMATED CHECKPOINT SWEEP & LEADERBOARD        [/bold cyan]")
    console.print(f"[bold cyan]================================================================[/bold cyan]")
    console.print(f"  Checkpoints Found: {len(ckpts)} ({', '.join(c.name for c in ckpts)})")
    console.print(f"  Test Samples:      {args.samples} holdout samples per checkpoint\n")

    # Load holdout samples
    sft_data_path = Path("data/sft_dataset.jsonl")
    rows = []
    with open(sft_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    test_samples = rows[-args.samples:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_id = "Qwen/Qwen2.5-1.5B-Instruct"

    console.print(f"[yellow]⏳ Loading Base Tokenizer: {base_id}...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    leaderboard = []

    for i, ckpt in enumerate(ckpts, 1):
        step_num = int(ckpt.name.split("-")[-1]) if ckpt.name.split("-")[-1].isdigit() else 0
        console.print(f"\n[bold]Evaluating [{i}/{len(ckpts)}] {ckpt.name} (Step {step_num})...[/bold]")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            base_model = base_model.to("cpu")

        peft_model = PeftModel.from_pretrained(base_model, str(ckpt))
        try:
            model = peft_model.merge_and_unload()
        except Exception:
            model = peft_model
        model.eval()

        results = []
        for sample in track(test_samples, description=f"  Testing {ckpt.name}..."):
            convs = sample["conversations"]
            expected = [c["content"] for c in convs if c["role"] == "assistant"][0]
            schema_keys = extract_schema_keys(convs[0]["content"])
            res = evaluate_single(model, tokenizer, convs, expected, schema_keys)
            results.append(res)

        N = len(results)
        json_rate = sum(1 for r in results if r["valid_json"]) / N * 100
        clean_rate = sum(1 for r in results if r["clean_format"]) / N * 100
        tool_acc = sum(1 for r in results if r["tool_match"]) / N * 100
        param_acc = sum(1 for r in results if r["param_match"]) / N * 100
        halluc_rate = sum(1 for r in results if r["no_hallucination"]) / N * 100
        avg_lat = sum(r["latency_ms"] for r in results) / N

        # Overall composite score (Higher is better)
        overall_score = (tool_acc * 0.4) + (param_acc * 0.4) + (json_rate * 0.1) + (clean_rate * 0.1)

        leaderboard.append({
            "checkpoint": ckpt.name,
            "path": str(ckpt),
            "step": step_num,
            "overall_score": overall_score,
            "tool_acc": tool_acc,
            "param_acc": param_acc,
            "json_rate": json_rate,
            "clean_rate": clean_rate,
            "halluc_rate": halluc_rate,
            "latency_ms": avg_lat,
        })

        del model, base_model, peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Sort leaderboard by overall score descending
    leaderboard.sort(key=lambda x: x["overall_score"], reverse=True)

    # Display Leaderboard Table
    table = Table(title="🏆 Sift-1B Checkpoint Sweep Leaderboard 🏆", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center", style="bold yellow", width=6)
    table.add_column("Checkpoint", style="cyan", width=18)
    table.add_column("Overall Score", justify="right", style="bold green", width=16)
    table.add_column("Tool Acc", justify="right", width=12)
    table.add_column("Param Acc", justify="right", width=12)
    table.add_column("JSON Rate", justify="right", width=12)
    table.add_column("Clean Format", justify="right", width=14)
    table.add_column("Avg Latency", justify="right", width=14)

    for rank, entry in enumerate(leaderboard, 1):
        medal = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else f"#{rank} "))
        table.add_row(
            f"{medal}",
            entry["checkpoint"],
            f"{entry['overall_score']:.1f}%",
            f"{entry['tool_acc']:.1f}%",
            f"{entry['param_acc']:.1f}%",
            f"{entry['json_rate']:.1f}%",
            f"{entry['clean_rate']:.1f}%",
            f"{entry['latency_ms']:.0f} ms",
        )

    console.print("\n")
    console.print(table)

    winner = leaderboard[0]
    console.print(f"\n[bold green]================================================================[/bold green]")
    console.print(f"[bold green]🏆 THE WINNER IS: {winner['checkpoint']} (Score: {winner['overall_score']:.1f}%)[/bold green]")
    console.print(f"  Tool Accuracy:       {winner['tool_acc']:.1f}%")
    console.print(f"  Param Accuracy:      {winner['param_acc']:.1f}%")
    console.print(f"  JSON Validity:       {winner['json_rate']:.1f}%")
    console.print(f"  Clean Formatting:    {winner['clean_rate']:.1f}%")
    console.print(f"  Average Latency:     {winner['latency_ms']:.0f} ms")
    console.print(f"\n[bold cyan]Deploy with:[/bold cyan]")
    console.print(f"  python export/push_dpo.py --source {winner['path']}")
    console.print(f"[bold green]================================================================[/bold green]\n")

    # Save to JSON
    sweep_path = Path("eval/checkpoint_sweep_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)


if __name__ == "__main__":
    main()
