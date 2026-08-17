"""
Sift — DPO Preference Dataset Generator
=========================================
The "secret sauce" of the Sift training pipeline.

Takes the clean SFT dataset (data/sft_dataset.jsonl) and programmatically
generates "rejected" responses by applying 5 mutation strategies that
simulate real-world LLM failure modes.

The DPO trainer will learn to HEAVILY penalize these failure patterns,
producing a model that outputs strict, type-safe JSON on the first attempt.

Output: data/dpo_dataset.jsonl

Mutation Strategies:
  1. Markdown Wrapper    — Wraps JSON in ```json``` with conversational preamble
  2. Type Coercion       — Converts int→str, bool→str, float→int
  3. Hallucinated Key    — Injects a fake parameter not in the schema
  4. Missing Required Key — Deletes a required parameter
  5. Argument Swap       — Swaps values between two parameters
"""

import json
import copy
import random
import argparse
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

SEED = 42
PREAMBLES = [
    "Sure! Here is the function call you requested:\n",
    "Certainly! Let me help you with that.\n\n",
    "Of course! I'll process that right away.\n",
    "Here is what I've prepared for you:\n",
    "Great question! Here's the result:\n\n",
    "I'd be happy to help! Here's the function call:\n",
    "Let me route that for you:\n\n",
    "Based on your request, here is the output:\n",
]

HALLUCINATED_KEYS = [
    ("confidence_score", 0.95),
    ("request_id", "req_abc123"),
    ("processing_time_ms", 42),
    ("model_version", "v2.1"),
    ("cache_hit", True),
    ("metadata", {"source": "internal"}),
    ("priority", "high"),
    ("timestamp", "2025-01-01T00:00:00Z"),
    ("debug_info", "processed_successfully"),
    ("retry_count", 0),
]


# =============================================================================
# Mutation Functions
# =============================================================================

def mutate_markdown_wrapper(answer_json: str) -> str:
    """
    Mutation 1: Wrap the valid JSON in markdown code fences and
    prepend conversational fluff. This is the #1 most common failure
    mode of general-purpose LLMs when asked for structured output.
    """
    preamble = random.choice(PREAMBLES)
    # Randomly choose between ```json and ``` wrapping
    if random.random() < 0.7:
        return f"{preamble}```json\n{answer_json}\n```"
    else:
        return f"{preamble}{answer_json}\n\nLet me know if you need anything else!"


def mutate_type_coercion(answer_json: str) -> str:
    """
    Mutation 2: Randomly convert numeric/boolean values to strings.
    e.g., "age": 25 → "age": "25"  or  "active": true → "active": "true"
    
    This teaches the model to preserve exact types from the schema.
    """
    try:
        data = json.loads(answer_json)
    except json.JSONDecodeError:
        return answer_json

    def coerce_values(obj):
        if isinstance(obj, dict):
            mutated = {}
            for k, v in obj.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and random.random() < 0.5:
                    mutated[k] = str(v)
                elif isinstance(v, bool) and random.random() < 0.5:
                    mutated[k] = str(v).lower()
                elif isinstance(v, dict):
                    mutated[k] = coerce_values(v)
                elif isinstance(v, list):
                    mutated[k] = [coerce_values(item) if isinstance(item, dict) else item for item in v]
                else:
                    mutated[k] = v
            return mutated
        return obj

    if isinstance(data, list):
        mutated_data = [coerce_values(item) for item in data]
    else:
        mutated_data = coerce_values(data)

    result = json.dumps(mutated_data, separators=(",", ":"))

    # Ensure we actually mutated something — if not, force a mutation
    if result == answer_json:
        return mutate_markdown_wrapper(answer_json)

    return result


def mutate_hallucinated_key(answer_json: str) -> str:
    """
    Mutation 3: Inject a completely fake parameter that was NOT in the
    tool schema. This teaches the model to never add extra fields.
    """
    try:
        data = json.loads(answer_json)
    except json.JSONDecodeError:
        return answer_json

    fake_key, fake_value = random.choice(HALLUCINATED_KEYS)

    def inject_key(obj):
        if isinstance(obj, dict):
            # Look for the 'arguments' or 'parameters' dict to inject into
            if "arguments" in obj and isinstance(obj["arguments"], dict):
                obj["arguments"][fake_key] = fake_value
                return obj
            elif "parameters" in obj and isinstance(obj["parameters"], dict):
                obj["parameters"][fake_key] = fake_value
                return obj
            else:
                # Inject at the top level of the dict
                obj[fake_key] = fake_value
                return obj
        return obj

    if isinstance(data, list) and len(data) > 0:
        # Inject into a random function call in the list
        target_idx = random.randint(0, len(data) - 1)
        data[target_idx] = inject_key(data[target_idx])
    else:
        data = inject_key(data)

    return json.dumps(data, separators=(",", ":"))


def mutate_missing_required_key(answer_json: str) -> str:
    """
    Mutation 4: Delete a parameter from the function call arguments.
    This teaches the model to always include all required fields.
    """
    try:
        data = json.loads(answer_json)
    except json.JSONDecodeError:
        return answer_json

    def remove_key(obj):
        if isinstance(obj, dict):
            # Find the arguments dict
            args_dict = None
            args_key = None
            if "arguments" in obj and isinstance(obj["arguments"], dict):
                args_dict = obj["arguments"]
                args_key = "arguments"
            elif "parameters" in obj and isinstance(obj["parameters"], dict):
                args_dict = obj["parameters"]
                args_key = "parameters"

            if args_dict and len(args_dict) > 1:
                # Remove a random key from arguments
                key_to_remove = random.choice(list(args_dict.keys()))
                del args_dict[key_to_remove]
                obj[args_key] = args_dict
                return obj, True
            elif args_dict and len(args_dict) == 1:
                # If only one key, just clear it
                obj[args_key] = {}
                return obj, True

        return obj, False

    if isinstance(data, list) and len(data) > 0:
        target_idx = random.randint(0, len(data) - 1)
        data[target_idx], mutated = remove_key(data[target_idx])
        if not mutated:
            return mutate_markdown_wrapper(answer_json)
    else:
        data, mutated = remove_key(data)
        if not mutated:
            return mutate_markdown_wrapper(answer_json)

    return json.dumps(data, separators=(",", ":"))


def mutate_argument_swap(answer_json: str) -> str:
    """
    Mutation 5: Swap values between two parameters in the same function call.
    e.g., {"city": "NYC", "country": "US"} → {"city": "US", "country": "NYC"}
    
    This teaches the model to correctly bind values to their parameters.
    """
    try:
        data = json.loads(answer_json)
    except json.JSONDecodeError:
        return answer_json

    def swap_values(obj):
        if isinstance(obj, dict):
            args_dict = None
            args_key = None
            if "arguments" in obj and isinstance(obj["arguments"], dict):
                args_dict = obj["arguments"]
                args_key = "arguments"
            elif "parameters" in obj and isinstance(obj["parameters"], dict):
                args_dict = obj["parameters"]
                args_key = "parameters"

            if args_dict and len(args_dict) >= 2:
                keys = list(args_dict.keys())
                k1, k2 = random.sample(keys, 2)
                args_dict[k1], args_dict[k2] = args_dict[k2], args_dict[k1]
                obj[args_key] = args_dict
                return obj, True

        return obj, False

    if isinstance(data, list) and len(data) > 0:
        target_idx = random.randint(0, len(data) - 1)
        data[target_idx], swapped = swap_values(data[target_idx])
        if not swapped:
            return mutate_markdown_wrapper(answer_json)
    else:
        data, swapped = swap_values(data)
        if not swapped:
            return mutate_markdown_wrapper(answer_json)

    return json.dumps(data, separators=(",", ":"))


# =============================================================================
# Mutation Registry
# =============================================================================

MUTATIONS: list[tuple[str, Callable[[str], str], float]] = [
    ("markdown_wrapper",    mutate_markdown_wrapper,      0.30),  # 30% — most common failure
    ("type_coercion",       mutate_type_coercion,         0.20),  # 20%
    ("hallucinated_key",    mutate_hallucinated_key,      0.20),  # 20%
    ("missing_required",    mutate_missing_required_key,  0.15),  # 15%
    ("argument_swap",       mutate_argument_swap,         0.15),  # 15%
]


def select_mutation() -> tuple[str, Callable[[str], str]]:
    """Select a mutation strategy based on weighted probabilities."""
    names, funcs, weights = zip(*MUTATIONS)
    chosen = random.choices(list(zip(names, funcs)), weights=weights, k=1)[0]
    return chosen


def generate_rejected(answer_json: str) -> tuple[str, str]:
    """
    Generate a single rejected response by applying a random mutation.
    Returns (mutation_name, rejected_text).
    """
    mutation_name, mutation_fn = select_mutation()
    rejected = mutation_fn(answer_json)
    return mutation_name, rejected


# =============================================================================
# Main Pipeline
# =============================================================================

def main(input_path: str = None, output_path: str = None):
    """
    Main pipeline: Load SFT data → Generate rejected pairs → Save DPO dataset
    """
    if input_path is None:
        input_path = str(Path(__file__).parent / "sft_dataset.jsonl")
    if output_path is None:
        output_path = str(Path(__file__).parent / "dpo_dataset.jsonl")

    console.print(f"\n[bold cyan]Sift — DPO Preference Dataset Generator[/bold cyan]")
    console.print(f"  Input:  {input_path}")
    console.print(f"  Output: {output_path}\n")

    random.seed(SEED)

    # -------------------------------------------------------------------------
    # Step 1: Load the clean SFT dataset
    # -------------------------------------------------------------------------
    input_file = Path(input_path)
    if not input_file.exists():
        console.print(f"[red]✗ SFT dataset not found at {input_path}[/red]")
        console.print("[yellow]  Run `python data/prepare_sft.py` first.[/yellow]")
        return

    sft_rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sft_rows.append(json.loads(line))

    console.print(f"[green]✓ Loaded {len(sft_rows):,} SFT rows[/green]")

    # -------------------------------------------------------------------------
    # Step 2: Generate DPO preference pairs
    # -------------------------------------------------------------------------
    dpo_rows = []
    mutation_counts = {name: 0 for name, _, _ in MUTATIONS}
    skipped = 0

    for row in track(sft_rows, description="Generating DPO pairs..."):
        conversations = row.get("conversations", [])

        # Extract the prompt (system + user) and the chosen answer (assistant)
        prompt_messages = []
        chosen_message = None

        for msg in conversations:
            if msg["role"] in ("system", "user"):
                prompt_messages.append(msg)
            elif msg["role"] == "assistant":
                chosen_message = msg

        if not prompt_messages or chosen_message is None:
            skipped += 1
            continue

        # The chosen response is the clean, valid JSON
        chosen_text = chosen_message["content"]

        # Generate the rejected response via mutation
        mutation_name, rejected_text = generate_rejected(chosen_text)
        mutation_counts[mutation_name] += 1

        # Ensure rejected is actually different from chosen
        if rejected_text == chosen_text:
            rejected_text = mutate_markdown_wrapper(chosen_text)
            mutation_counts["markdown_wrapper"] += 1
            mutation_counts[mutation_name] -= 1

        # Build the DPO row in TRL-compatible format
        dpo_row = {
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
        }
        dpo_rows.append(dpo_row)

    console.print(f"[green]✓ Generated {len(dpo_rows):,} preference pairs[/green]")
    if skipped > 0:
        console.print(f"[yellow]⚠ Skipped {skipped} malformed rows[/yellow]")

    # -------------------------------------------------------------------------
    # Step 3: Save as JSONL
    # -------------------------------------------------------------------------
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dpo_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    console.print(f"[bold green]✓ Saved to {output_path} ({file_size_mb:.1f} MB)[/bold green]")

    # -------------------------------------------------------------------------
    # Step 4: Print mutation distribution
    # -------------------------------------------------------------------------
    console.print("\n[bold]Mutation Distribution:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Mutation Strategy", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total = sum(mutation_counts.values())
    for name, count in sorted(mutation_counts.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        table.add_row(name, f"{count:,}", f"{pct:.1f}%")

    console.print(table)
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO preference dataset for Sift")
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input SFT dataset path (default: data/sft_dataset.jsonl)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output DPO dataset path (default: data/dpo_dataset.jsonl)"
    )
    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output)
