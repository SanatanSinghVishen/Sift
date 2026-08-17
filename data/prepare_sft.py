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

DATASET_ID = "Salesforce/xlam-function-calling-60k"
DEFAULT_SAMPLE_SIZE = 10_000
SEED = 42

SYSTEM_PROMPT = (
    "You are a strict function calling agent. You are provided with a list of "
    "available tools in JSON format. Based on the user's request, you MUST "
    "output ONLY a valid JSON array of function calls. Do NOT include any "
    "explanation, markdown formatting, or conversational text. Output raw JSON only."
)


def format_tools_for_prompt(tools_raw: str) -> str:
    """
    Parse the raw tools JSON string and re-serialize it cleanly.
    The xlam dataset stores tools as a JSON string that may contain
    nested stringified JSON — we normalize it here.
    """
    try:
        tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        return json.dumps(tools, indent=2)
    except (json.JSONDecodeError, TypeError):
        return tools_raw


def format_answer(answers_raw: str) -> str:
    """
    Parse the raw answers JSON string and re-serialize it as compact JSON.
    This is the 'ground truth' function call the model should learn to produce.
    """
    try:
        answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        return json.dumps(answers, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return answers_raw


def transform_row(row: dict) -> dict | None:
    """
    Transform a single xlam dataset row into a ChatML conversation.

    Returns None if the row is malformed (missing fields, unparseable JSON).
    """
    query = row.get("query", "").strip()
    tools_raw = row.get("tools", "")
    answers_raw = row.get("answers", "")

    # Skip rows with missing data
    if not query or not tools_raw or not answers_raw:
        return None

    # Format the tools into readable JSON for the system prompt
    tools_formatted = format_tools_for_prompt(tools_raw)

    # Format the answer as compact JSON (what the model should output)
    answer_formatted = format_answer(answers_raw)

    # Validate that the answer is actually valid JSON
    try:
        json.loads(answer_formatted)
    except (json.JSONDecodeError, TypeError):
        return None

    # Build the system message with embedded tool definitions
    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{tools_formatted}"
    )

    # Construct the ChatML conversation
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
    Main pipeline: Download → Sample → Transform → Validate → Save
    """
    if output_path is None:
        output_path = str(Path(__file__).parent / "sft_dataset.jsonl")

    console.print(f"\n[bold cyan]Sift — SFT Dataset Preparation[/bold cyan]")
    console.print(f"  Source:      {DATASET_ID}")
    console.print(f"  Sample size: {sample_size:,}")
    console.print(f"  Output:      {output_path}\n")

    # -------------------------------------------------------------------------
    # Step 1: Stream or Download dataset
    # -------------------------------------------------------------------------
    console.print("[yellow]⏳ Streaming dataset from Hugging Face (fetching only required rows)...[/yellow]")
    import os
    hf_token = os.environ.get("HF_TOKEN") or None

    try:
        # Using streaming=True fetches rows on the fly in seconds
        # without downloading the entire 60k dataset to disk!
        streamed_dataset = load_dataset(
            DATASET_ID,
            split="train",
            streaming=True,
            token=hf_token
        )
    except Exception as e:
        console.print(f"\n[red]✗ Failed to access dataset: {e}[/red]\n")
        console.print("[yellow]Notice: 'Salesforce/xlam-function-calling-60k' is a gated dataset on Hugging Face.[/yellow]")
        console.print("[cyan]Quick 2-step fix:[/cyan]")
        console.print("  1. Open [bold]https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k[/bold] in your browser and click '[bold]Agree and access repository[/bold]'.")
        console.print("  2. In your terminal, run:")
        console.print("       [bold green]huggingface-cli login[/bold green]")
        console.print("     (or set: [bold green]export HF_TOKEN=\"hf_your_token_here\"[/bold green])\n")
        return

    # -------------------------------------------------------------------------
    # Step 2: Stream, transform & sample rows on the fly
    # -------------------------------------------------------------------------
    transformed = []
    skipped = 0
    total_processed = 0

    # Shuffle buffer to get diverse samples while streaming
    shuffled_stream = streamed_dataset.shuffle(seed=SEED, buffer_size=1000)

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Streaming and formatting rows...", total=sample_size)

        for row in shuffled_stream:
            total_processed += 1
            result = transform_row(row)
            if result is not None:
                transformed.append(result)
                progress.advance(task)
                if len(transformed) >= sample_size:
                    break
            else:
                skipped += 1

    console.print(f"[green]✓ Successfully prepared {len(transformed):,} rows (processed {total_processed:,})[/green]")
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
