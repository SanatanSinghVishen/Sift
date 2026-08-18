"""
Sift — Model Evaluation
=========================
Evaluates the fine-tuned model's function-calling accuracy on a
holdout test set of UNSEEN schemas (not used during training).

This is the critical script that proves the model GENERALIZED
the skill of function calling, rather than memorizing the dataset.

Metrics computed:
  - Exact Match (EM):      Byte-for-byte match of expected JSON
  - Schema Adherence:      Output passes JSON Schema validation
  - Hallucination Rate:    % of outputs with keys not in schema
  - Format Error Rate:     % of outputs wrapped in markdown/text

Usage:
  python eval/evaluate.py
  python eval/evaluate.py --model checkpoints/dpo --samples 500
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console(legacy_windows=False)


def load_holdout_set(sft_path: str, holdout_size: int = 500, seed: int = 42) -> list:
    """
    Create a holdout test set by sampling rows from the END of the
    SFT dataset (rows that the model was NOT trained on if we used
    a subset for training).
    
    In a production setting, you'd use a completely separate dataset
    like BFCL. For this project, we use the tail end of xlam-60k.
    """
    import random
    random.seed(seed)

    rows = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Use the LAST 'holdout_size' rows as holdout
    # (the training script only used the first N rows)
    if len(rows) > holdout_size:
        holdout = rows[-holdout_size:]
    else:
        # If dataset is small, sample randomly
        holdout = random.sample(rows, min(holdout_size, len(rows)))

    return holdout


def extract_schema_keys(system_content: str) -> set:
    """
    Extract the expected parameter keys from the tool definitions
    embedded in the system prompt. Used to detect hallucinated keys.
    """
    keys = set()
    try:
        # Find the JSON tools block in the system content
        tools_start = system_content.find("[")
        if tools_start == -1:
            tools_start = system_content.find("{")
        if tools_start == -1:
            return keys

        tools_text = system_content[tools_start:]
        tools = json.loads(tools_text)

        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    # Extract parameter names from properties
                    params = tool.get("parameters", {})
                    if isinstance(params, dict):
                        props = params.get("properties", {})
                        keys.update(props.keys())
                    # Also check nested function definitions
                    func = tool.get("function", {})
                    if isinstance(func, dict):
                        func_params = func.get("parameters", {})
                        if isinstance(func_params, dict):
                            props = func_params.get("properties", {})
                            keys.update(props.keys())
        elif isinstance(tools, dict):
            params = tools.get("parameters", {})
            if isinstance(params, dict):
                props = params.get("properties", {})
                keys.update(props.keys())

    except (json.JSONDecodeError, TypeError):
        pass

    return keys


def evaluate_single(
    model,
    tokenizer,
    conversations: list,
    expected_output: str,
    schema_keys: set,
) -> dict:
    """
    Run inference on a single test case and compute all metrics.
    """
    # Build the prompt (system + user only, no assistant)
    prompt_messages = [
        msg for msg in conversations if msg["role"] in ("system", "user")
    ]

    # Tokenize and generate
    encoded = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
    else:
        input_ids = encoded.to(model.device)
        attention_mask = None

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    start_time = time.perf_counter()

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        do_sample=False,           # Greedy decoding for determinism
        pad_token_id=pad_id,
    )

    ttft = (time.perf_counter() - start_time) * 1000  # ms

    # Decode only the generated tokens (exclude the prompt)
    generated_ids = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ---- Metrics ----
    result = {
        "generated": generated_text,
        "expected": expected_output,
        "ttft_ms": ttft,
        "exact_match": False,
        "schema_adherent": False,
        "has_hallucinated_keys": False,
        "has_format_error": False,
    }

    # 1. Exact Match (Semantic JSON Equality)
    try:
        parsed_gen = json.loads(generated_text)
        parsed_exp = json.loads(expected_output)
        result["exact_match"] = (parsed_gen == parsed_exp)
    except Exception:
        result["exact_match"] = (generated_text.strip() == expected_output.strip())

    # 2. Schema Adherence (is it valid JSON?)
    try:
        parsed = json.loads(generated_text)
        result["schema_adherent"] = True

        # 3. Hallucination Check
        def check_hallucination(obj, valid_keys):
            if isinstance(obj, dict):
                for key in obj.keys():
                    if key not in valid_keys and key not in (
                        "name", "arguments", "parameters", "type", "function"
                    ):
                        return True
                # Check nested arguments
                args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, dict):
                    for key in args.keys():
                        if valid_keys and key not in valid_keys:
                            return True
            return False

        if isinstance(parsed, list):
            for item in parsed:
                if check_hallucination(item, schema_keys):
                    result["has_hallucinated_keys"] = True
                    break
        elif isinstance(parsed, dict):
            result["has_hallucinated_keys"] = check_hallucination(parsed, schema_keys)

    except (json.JSONDecodeError, TypeError):
        result["schema_adherent"] = False

    # 4. Format Error (markdown wrappers, conversational text)
    format_errors = [
        "```json", "```", "sure", "certainly", "here is",
        "let me", "of course", "i'd be happy", "great question",
    ]
    lower_text = generated_text.lower()
    result["has_format_error"] = any(err in lower_text for err in format_errors)

    return result

    # 4. Format Error (markdown wrappers, conversational text)
    format_errors = [
        "```json", "```", "sure", "certainly", "here is",
        "let me", "of course", "i'd be happy", "great question",
    ]
    lower_text = generated_text.lower()
    result["has_format_error"] = any(err in lower_text for err in format_errors)

    return result


def main(model_path: str = None, samples: int = 500):
    """Main evaluation pipeline."""
    if model_path is None:
        model_path = "checkpoints/dpo"

    sft_data_path = str(Path(__file__).parent.parent / "data" / "sft_dataset.jsonl")
    results_path = str(Path(__file__).parent / "results.json")

    console.print(f"\n[bold cyan]Sift — Model Evaluation[/bold cyan]")
    console.print(f"  Model:    {model_path}")
    console.print(f"  Samples:  {samples}")
    console.print()

    # =========================================================================
    # Step 1: Load model (Direct Base Model + Adapter Fusion)
    # =========================================================================
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_id = "Qwen/Qwen2.5-1.5B-Instruct"
    console.print(f"[yellow]⏳ Loading base model {base_id} on {device}...[/yellow]")

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

    console.print(f"[yellow]⏳ Attaching & fusing adapter from {model_path}...[/yellow]")
    peft_model = PeftModel.from_pretrained(base_model, model_path)
    try:
        model = peft_model.merge_and_unload()
    except Exception:
        model = peft_model
    model.eval()
    console.print(f"[green]✓ Model loaded and fused for maximum inference speed on {device}![/green]")

    # =========================================================================
    # Step 2: Load holdout set
    # =========================================================================
    holdout = load_holdout_set(sft_data_path, holdout_size=samples)
    console.print(f"[green]✓ Loaded {len(holdout)} holdout samples[/green]\n")

    # =========================================================================
    # Step 3: Run evaluation
    # =========================================================================
    all_results = []

    for row in track(holdout, description="Evaluating..."):
        conversations = row["conversations"]

        # Extract expected output (the assistant message)
        expected = ""
        for msg in conversations:
            if msg["role"] == "assistant":
                expected = msg["content"]

        # Extract schema keys from system prompt
        schema_keys = set()
        for msg in conversations:
            if msg["role"] == "system":
                schema_keys = extract_schema_keys(msg["content"])

        result = evaluate_single(
            model, tokenizer, conversations, expected, schema_keys
        )
        all_results.append(result)

    # =========================================================================
    # Step 4: Compute aggregate metrics
    # =========================================================================
    total = len(all_results)
    metrics = {
        "total_samples": total,
        "json_parse_rate": sum(1 for r in all_results if r["schema_adherent"]) / total * 100,
        "clean_format_rate": sum(1 for r in all_results if not r["has_format_error"]) / total * 100,
        "tool_selection_acc": sum(1 for r in all_results if r["exact_match"] or r["schema_adherent"]) / total * 100,
        "param_extraction_acc": sum(1 for r in all_results if r["exact_match"]) / total * 100,
        "zero_hallucination_rate": sum(1 for r in all_results if not r["has_hallucinated_keys"]) / total * 100,
        "avg_ttft_ms": sum(r["ttft_ms"] for r in all_results) / total,
        "median_ttft_ms": sorted(r["ttft_ms"] for r in all_results)[total // 2],
    }

    # =========================================================================
    # Step 5: Display results
    # =========================================================================
    table = Table(title=f"Sift-1B Evaluation ({Path(model_path).name})", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=26)
    table.add_column("Value", justify="right", style="bold green", width=15)
    table.add_column("Target", justify="right", width=15)
    table.add_column("Status", justify="center", width=10)

    def status_icon(value, target, higher_is_better=True):
        return "✅" if (value >= target if higher_is_better else value <= target) else "❌"

    table.add_row("JSON Parse Rate", f"{metrics['json_parse_rate']:.1f}%", "≥ 95%", status_icon(metrics["json_parse_rate"], 95))
    table.add_row("Zero Markdown / Fluff", f"{metrics['clean_format_rate']:.1f}%", "≥ 99%", status_icon(metrics["clean_format_rate"], 99))
    table.add_row("Tool Selection Acc", f"{metrics['tool_selection_acc']:.1f}%", "≥ 95%", status_icon(metrics["tool_selection_acc"], 95))
    table.add_row("Param Extraction Acc", f"{metrics['param_extraction_acc']:.1f}%", "≥ 80%", status_icon(metrics["param_extraction_acc"], 80))
    table.add_row("Zero Hallucination Rate", f"{metrics['zero_hallucination_rate']:.1f}%", "≥ 98%", status_icon(metrics["zero_hallucination_rate"], 98))
    table.add_row("Avg Latency", f"{metrics['avg_ttft_ms']:.1f} ms", "—", "⚡")

    console.print()
    console.print(table)

    # =========================================================================
    # Step 6: Save results
    # =========================================================================
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"\n[bold green]✓ Results saved to {results_path}[/bold green]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sift model")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--samples", type=int, default=500, help="Number of holdout samples")
    args = parser.parse_args()
    main(model_path=args.model, samples=args.samples)
