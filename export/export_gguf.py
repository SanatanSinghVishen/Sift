"""
Sift — GGUF Export Script
==========================
Merges LoRA adapters into the base model and exports to GGUF format
for deployment via llama.cpp, Ollama, or LM Studio.

Usage:
  python export/export_gguf.py
  python export/export_gguf.py --model checkpoints/dpo --quantization q4_k_m
"""

import argparse
from pathlib import Path
from rich.console import Console

console = Console()


def main(
    model_path: str = None,
    quantization: str = "q4_k_m",
    output_dir: str = None,
    push_to_hub: bool = False,
    hub_repo: str = "SanatanSinghVishen/sift-1b",
):
    """Export the fine-tuned model to GGUF format."""
    if model_path is None:
        model_path = "checkpoints/dpo"
    if output_dir is None:
        output_dir = "export/sift-1b"

    console.print(f"\n[bold cyan]Sift — GGUF Export[/bold cyan]")
    console.print(f"  Model:         {model_path}")
    console.print(f"  Quantization:  {quantization}")
    console.print(f"  Output:        {output_dir}")
    console.print(f"  Push to Hub:   {push_to_hub}")
    console.print()

    # =========================================================================
    # Step 1: Load the trained model
    # =========================================================================
    console.print("[yellow]⏳ Loading model...[/yellow]")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        load_in_4bit=True,
    )

    console.print("[green]✓ Model loaded[/green]")

    # =========================================================================
    # Step 2: Save to GGUF locally
    # =========================================================================
    console.print(f"[yellow]⏳ Exporting to GGUF ({quantization})...[/yellow]")
    console.print("[dim]This may take a few minutes...[/dim]")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quantization,
    )

    console.print(f"[bold green]✓ GGUF saved to {output_dir}/[/bold green]")

    # =========================================================================
    # Step 3: Also save LoRA adapters (SafeTensors) for fine-tuning reuse
    # =========================================================================
    lora_dir = f"{output_dir}/lora"
    Path(lora_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    console.print(f"[green]✓ LoRA adapters saved to {lora_dir}/[/green]")

    # =========================================================================
    # Step 4: Push to Hugging Face (optional)
    # =========================================================================
    if push_to_hub:
        console.print(f"\n[yellow]⏳ Pushing to Hugging Face: {hub_repo}[/yellow]")

        # Push GGUF
        model.push_to_hub_gguf(
            hub_repo,
            tokenizer,
            quantization_method=quantization,
        )

        # Push LoRA adapters
        model.push_to_hub(hub_repo + "-lora", tokenizer)

        console.print(f"[bold green]✓ Pushed to https://huggingface.co/{hub_repo}[/bold green]")

    # =========================================================================
    # Summary
    # =========================================================================
    console.print(f"\n[bold]Export complete! Test locally with:[/bold]")
    console.print(f"  python -m llama_cpp.server --model {output_dir}/unsloth.Q4_K_M.gguf --n_gpu_layers -1")
    console.print(f"\n  Or with Ollama:")
    console.print(f"  ollama create sift-1b -f {output_dir}/Modelfile")
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Sift model to GGUF")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument(
        "--quantization", type=str, default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16", "q5_k_m", "q4_0"],
        help="GGUF quantization method"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument(
        "--hub-repo", type=str, default="SanatanSinghVishen/sift-1b",
        help="Hugging Face repository name"
    )
    args = parser.parse_args()
    main(
        model_path=args.model,
        quantization=args.quantization,
        output_dir=args.output,
        push_to_hub=args.push,
        hub_repo=args.hub_repo,
    )
