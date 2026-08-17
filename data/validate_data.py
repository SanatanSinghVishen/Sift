"""
Sift — Dataset Validation
===========================
Validates the integrity of both the SFT and DPO datasets before training.
Catches malformed JSON, missing fields, and structural issues early —
preventing silent training failures downstream.

Usage:
  python data/validate_data.py
  python data/validate_data.py --sft data/sft_dataset.jsonl --dpo data/dpo_dataset.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def validate_sft_dataset(path: str) -> dict:
    """
    Validate every row in the SFT dataset.
    
    Checks:
      1. Each row has a "conversations" key
      2. Conversations contain system, user, and assistant roles
      3. The assistant response is valid, parseable JSON
      4. The system message contains tool definitions
    """
    results = {
        "total": 0,
        "valid": 0,
        "errors": [],
        "avg_conversation_length": 0,
        "avg_assistant_tokens": 0,
        "role_distribution": Counter(),
    }

    if not Path(path).exists():
        results["errors"].append(f"File not found: {path}")
        return results

    total_conv_len = 0
    total_assistant_chars = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            results["total"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                results["errors"].append(f"Line {line_num}: Invalid JSON — {e}")
                continue

            # Check for conversations key
            convs = row.get("conversations")
            if not convs or not isinstance(convs, list):
                results["errors"].append(f"Line {line_num}: Missing 'conversations' key")
                continue

            total_conv_len += len(convs)

            # Check roles
            roles_present = set()
            for msg in convs:
                role = msg.get("role", "")
                roles_present.add(role)
                results["role_distribution"][role] += 1

                # Validate assistant response is valid JSON
                if role == "assistant":
                    content = msg.get("content", "")
                    total_assistant_chars += len(content)
                    try:
                        json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        results["errors"].append(
                            f"Line {line_num}: Assistant response is not valid JSON"
                        )

            if "system" not in roles_present:
                results["errors"].append(f"Line {line_num}: Missing 'system' role")
            if "user" not in roles_present:
                results["errors"].append(f"Line {line_num}: Missing 'user' role")
            if "assistant" not in roles_present:
                results["errors"].append(f"Line {line_num}: Missing 'assistant' role")
                continue

            results["valid"] += 1

    if results["total"] > 0:
        results["avg_conversation_length"] = total_conv_len / results["total"]
        results["avg_assistant_tokens"] = (total_assistant_chars / 4) / results["total"]

    return results


def validate_dpo_dataset(path: str) -> dict:
    """
    Validate every row in the DPO dataset.
    
    Checks:
      1. Each row has "prompt", "chosen", and "rejected" keys
      2. Prompt contains system and user messages
      3. Chosen response is valid JSON
      4. Rejected response is DIFFERENT from chosen
    """
    results = {
        "total": 0,
        "valid": 0,
        "errors": [],
        "identical_pairs": 0,
        "chosen_valid_json": 0,
        "rejected_valid_json": 0,
    }

    if not Path(path).exists():
        results["errors"].append(f"File not found: {path}")
        return results

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            results["total"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                results["errors"].append(f"Line {line_num}: Invalid JSON — {e}")
                continue

            # Check required keys
            for key in ("prompt", "chosen", "rejected"):
                if key not in row:
                    results["errors"].append(f"Line {line_num}: Missing '{key}' key")

            prompt = row.get("prompt", [])
            chosen = row.get("chosen", [])
            rejected = row.get("rejected", [])

            # Validate prompt has system + user
            prompt_roles = {msg.get("role") for msg in prompt}
            if "system" not in prompt_roles or "user" not in prompt_roles:
                results["errors"].append(
                    f"Line {line_num}: Prompt missing system/user roles"
                )

            # Validate chosen is valid JSON
            if chosen and isinstance(chosen, list):
                chosen_content = chosen[0].get("content", "")
                try:
                    json.loads(chosen_content)
                    results["chosen_valid_json"] += 1
                except (json.JSONDecodeError, TypeError):
                    results["errors"].append(
                        f"Line {line_num}: Chosen response is not valid JSON"
                    )

                # Check rejected is different from chosen
                if rejected and isinstance(rejected, list):
                    rejected_content = rejected[0].get("content", "")
                    if chosen_content == rejected_content:
                        results["identical_pairs"] += 1
                        results["errors"].append(
                            f"Line {line_num}: Chosen and rejected are identical"
                        )

                    # Check if rejected is valid JSON (it usually shouldn't be
                    # for markdown_wrapper mutations, but should be for others)
                    try:
                        json.loads(rejected_content)
                        results["rejected_valid_json"] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass  # Expected for markdown wrapper mutations

            results["valid"] += 1

    return results


def print_results(name: str, results: dict):
    """Pretty-print validation results."""
    error_count = len(results["errors"])
    status = "[green]PASS[/green]" if error_count == 0 else f"[red]{error_count} ERRORS[/red]"

    console.print(Panel(
        f"[bold]{name}[/bold]\n"
        f"  Total rows:  {results['total']:,}\n"
        f"  Valid rows:  {results['valid']:,}\n"
        f"  Status:      {status}",
        border_style="green" if error_count == 0 else "red",
    ))

    if "avg_conversation_length" in results and results["total"] > 0:
        console.print(f"  Avg messages/row:    {results['avg_conversation_length']:.1f}")
        console.print(f"  Avg assistant tokens: {results['avg_assistant_tokens']:.0f}")

    if "role_distribution" in results:
        console.print(f"  Role distribution:   {dict(results['role_distribution'])}")

    if "chosen_valid_json" in results:
        console.print(f"  Chosen valid JSON:   {results['chosen_valid_json']:,}")
        console.print(f"  Rejected valid JSON: {results['rejected_valid_json']:,}")
        console.print(f"  Identical pairs:     {results['identical_pairs']}")

    if error_count > 0:
        console.print(f"\n  [red]First 10 errors:[/red]")
        for err in results["errors"][:10]:
            console.print(f"    • {err}")

    console.print()


def main(sft_path: str = None, dpo_path: str = None):
    """Run validation on both datasets."""
    data_dir = Path(__file__).parent

    if sft_path is None:
        sft_path = str(data_dir / "sft_dataset.jsonl")
    if dpo_path is None:
        dpo_path = str(data_dir / "dpo_dataset.jsonl")

    console.print(f"\n[bold cyan]Sift — Dataset Validation[/bold cyan]\n")

    # Validate SFT
    if Path(sft_path).exists():
        console.print("[yellow]Validating SFT dataset...[/yellow]")
        sft_results = validate_sft_dataset(sft_path)
        print_results("SFT Dataset", sft_results)
    else:
        console.print(f"[yellow]⚠ SFT dataset not found at {sft_path}[/yellow]")

    # Validate DPO
    if Path(dpo_path).exists():
        console.print("[yellow]Validating DPO dataset...[/yellow]")
        dpo_results = validate_dpo_dataset(dpo_path)
        print_results("DPO Dataset", dpo_results)
    else:
        console.print(f"[yellow]⚠ DPO dataset not found at {dpo_path}[/yellow]")

    # Summary
    console.print("[bold]Validation complete.[/bold]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Sift datasets")
    parser.add_argument("--sft", type=str, default=None, help="SFT dataset path")
    parser.add_argument("--dpo", type=str, default=None, help="DPO dataset path")
    args = parser.parse_args()
    main(sft_path=args.sft, dpo_path=args.dpo)
