"""
Sift — SFT Dataset Preparation
================================
Downloads Salesforce/xlam-function-calling-60k from Hugging Face,
samples 10,000 rows, and transforms them into ChatML-formatted
conversations compatible with Unsloth's SFTTrainer.

Output: data/sft_dataset.jsonl

Dataset columns (xlam-60k):
  - query:   User's natural language request
  - tools:   JSON string of available function definitions
  - answers: JSON string of expected function call(s)

Target format (ChatML for Qwen2.5):
  [
    {"role": "system", "content": "<system prompt with tool definitions>"},
    {"role": "user",   "content": "<user query>"},
    {"role": "assistant", "content": "<strict JSON function call>"}
  ]
"""

import json
import random
import argparse
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()

# =============================================================================
# Constants
# =============================================================================

DATASET_ID = "minpeter/xlam-function-calling-60k-parsed"
FALLBACK_DATASET_ID = "Salesforce/xlam-function-calling-60k"
DEFAULT_SAMPLE_SIZE = 10_000
SEED = 42

SYSTEM_PROMPT = (
    "You are a strict function calling agent. You are provided with a list of "
    "available tools in JSON format. Based on the user's request, you MUST "
    "output ONLY a valid JSON array of function calls. Do NOT include any "
    "explanation, markdown formatting, or conversational text. Output raw JSON only."
)


def format_tools_for_prompt(tools_raw) -> str:
    """
    Parse the tools field and re-serialize it cleanly.
    Handles both JSON string and parsed dictionary/list representations.
    """
    if isinstance(tools_raw, (dict, list)):
        return json.dumps(tools_raw, indent=2)
    try:
        tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        return json.dumps(tools, indent=2)
    except (json.JSONDecodeError, TypeError):
        return str(tools_raw)


def format_answer(answers_raw) -> str:
    """
    Parse the answers field and re-serialize it as compact JSON.
    """
    if isinstance(answers_raw, (dict, list)):
        return json.dumps(answers_raw, separators=(",", ":"))
    try:
        answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        return json.dumps(answers, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return str(answers_raw)


def transform_row(row: dict) -> dict | None:
    """
    Transform a single xlam dataset row into a ChatML conversation.
    Supports both `messages` + `tools` schema and raw `query` + `answers` schema.
    """
    # -------------------------------------------------------------------------
    # Format A: Pre-parsed messages format (minpeter/xlam-function-calling-60k-parsed)
    # -------------------------------------------------------------------------
    if "messages" in row and isinstance(row["messages"], (list, tuple)):
        query = ""
        answer_formatted = ""

        for msg in row["messages"]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "user":
                query = str(msg.get("content", "")).strip()
            elif role == "assistant":
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]
                    # Normalize tool calls into clean compact JSON
                    normalized = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            if isinstance(fn, dict) and "name" in fn:
                                args = fn.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except Exception:
                                        pass
                                normalized.append({"name": fn["name"], "arguments": args})
                            else:
                                normalized.append(tc)
                        else:
                            normalized.append(tc)
                    answer_formatted = json.dumps(normalized, separators=(",", ":"))
                elif msg.get("content"):
                    answer_formatted = format_answer(msg["content"])

        tools_raw = row.get("tools", "")
        if not query or not answer_formatted:
            return None

        tools_formatted = format_tools_for_prompt(tools_raw)

    # -------------------------------------------------------------------------
    # Format B: Raw query / tools / answers format (Salesforce/xlam-60k)
    # -------------------------------------------------------------------------
    else:
        query = str(row.get("query", "")).strip()
        tools_raw = row.get("tools", "")
        answers_raw = row.get("answers", "")

        if not query or not tools_raw or not answers_raw:
            return None

        tools_formatted = format_tools_for_prompt(tools_raw)
        answer_formatted = format_answer(answers_raw)

    # Ensure answer is valid JSON
    try:
        json.loads(answer_formatted)
    except (json.JSONDecodeError, TypeError):
        return None

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{tools_formatted}"
    )

    conversation = {
        "conversations": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer_formatted},
        ]
    }

    return conversation


def main(sample_size: int = DEFAULT_SAMPLE_SIZE, output_path: str = None):
    """
    Main pipeline: Fast Download → Sample → Transform → Validate → Save
    """
    if output_path is None:
        output_path = str(Path(__file__).parent / "sft_dataset.jsonl")

    console.print(f"\n[bold cyan]Sift — SFT Dataset Preparation[/bold cyan]")
    console.print(f"  Source:      {DATASET_ID} (Salesforce XLAM-60K Parquet)")
    console.print(f"  Sample size: {sample_size:,}")
    console.print(f"  Output:      {output_path}\n")

    # -------------------------------------------------------------------------
    # Step 1: Fast Download dataset (High-speed 26MB Parquet)
    # -------------------------------------------------------------------------
    console.print("[yellow]⏳ Downloading dataset (fast 26MB parquet)...[/yellow]")
    import os
    hf_token = os.environ.get("HF_TOKEN") or None

    try:
        dataset = load_dataset(DATASET_ID, split="train", token=hf_token)
        console.print(f"[green]✓ Downloaded {len(dataset):,} rows[/green]")
    except Exception:
        console.print(f"[yellow]Retrying with {FALLBACK_DATASET_ID}...[/yellow]")
        dataset = load_dataset(FALLBACK_DATASET_ID, split="train", token=hf_token)
        console.print(f"[green]✓ Downloaded {len(dataset):,} rows[/green]")

    # -------------------------------------------------------------------------
    # Step 2: Shuffle and sample
    # -------------------------------------------------------------------------
    random.seed(SEED)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]
    console.print(f"[green]✓ Sampled {len(sampled_indices):,} rows (seed={SEED})[/green]")

    # -------------------------------------------------------------------------
    # Step 3: Transform each row into ChatML format
    # -------------------------------------------------------------------------
    transformed = []
    skipped = 0

    for idx in track(sampled_indices, description="Formatting ChatML rows..."):
        row = dataset[idx]
        result = transform_row(row)
        if result is not None:
            transformed.append(result)
        else:
            skipped += 1

    console.print(f"[green]✓ Successfully transformed {len(transformed):,} rows[/green]")
    if skipped > 0:
        console.print(f"[yellow]⚠ Skipped {skipped} malformed rows[/yellow]")

    # -------------------------------------------------------------------------
    # Step 4: Save as JSONL
    # -------------------------------------------------------------------------
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in transformed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    console.print(f"[bold green]✓ Saved to {output_path} ({file_size_mb:.1f} MB)[/bold green]")

    # -------------------------------------------------------------------------
    # Step 5: Print summary statistics
    # -------------------------------------------------------------------------
    console.print("\n[bold]Dataset Summary:[/bold]")
    console.print(f"  Total rows:      {len(transformed):,}")
    console.print(f"  Skipped rows:    {skipped}")

    # Calculate average token length (rough estimate: 1 token ≈ 4 chars)
    total_chars = sum(
        len(json.dumps(item)) for item in transformed
    )
    avg_chars = total_chars / len(transformed) if transformed else 0
    console.print(f"  Avg chars/row:   {avg_chars:.0f} (~{avg_chars/4:.0f} tokens)")
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT dataset for Sift")
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of rows to sample (default: {DEFAULT_SAMPLE_SIZE:,})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: data/sft_dataset.jsonl)"
    )
    args = parser.parse_args()
    main(sample_size=args.sample_size, output_path=args.output)
