"""
Sift — GGUF Export Script
==========================
Merges LoRA adapters into the base Qwen model and exports to GGUF format
for deployment via llama.cpp, Ollama, or LM Studio.

Usage:
  python export/export_gguf.py
  python export/export_gguf.py --model /content/drive/MyDrive/sift_dpo_backup/checkpoint-750 --quantization q4_k_m
  python export/export_gguf.py --model /content/drive/MyDrive/sift_dpo_backup/checkpoint-750 --push --hub-repo SanatanSinghVishen/sift-1b-gguf
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console

console = Console(legacy_windows=False)


def main(
    model_path: str = None,
    quantization: str = "q4_k_m",
    output_dir: str = None,
    push_to_hub: bool = False,
    hub_repo: str = "SanatanSinghVishen/sift-1b-gguf",
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
    # Step 1: Load base model + merge LoRA adapter into full weights
    # =========================================================================
    console.print("[yellow]⏳ Loading base model + merging LoRA adapter...[/yellow]")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_id = "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"  Base model:  {base_id}")
    console.print(f"  Device:      {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in float16 for GGUF conversion (need full precision weights, not 4-bit)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    console.print(f"[yellow]⏳ Attaching adapter from {model_path}...[/yellow]")
    peft_model = PeftModel.from_pretrained(base_model, model_path)

    console.print("[yellow]⏳ Merging LoRA weights into base model (merge_and_unload)...[/yellow]")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    console.print("[green]✓ Adapter merged into base weights successfully![/green]")

    # =========================================================================
    # Step 2: Save merged model as full SafeTensors (required for GGUF conversion)
    # =========================================================================
    merged_dir = f"{output_dir}/merged_full"
    console.print(f"[yellow]⏳ Saving merged model to {merged_dir}...[/yellow]")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    console.print(f"[green]✓ Merged SafeTensors model saved to {merged_dir}/[/green]")

    # Free GPU memory before conversion
    del merged_model, peft_model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: Convert to GGUF using llama.cpp's convert script
    # =========================================================================
    console.print(f"\n[yellow]⏳ Converting to GGUF ({quantization})...[/yellow]")
    console.print("[dim]This may take a few minutes...[/dim]")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if llama.cpp convert script is available
    llama_cpp_dir = Path("/content/llama.cpp")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        console.print("[yellow]⏳ Cloning llama.cpp for GGUF conversion...[/yellow]")
        os.system("git clone --depth 1 https://github.com/ggerganov/llama.cpp /content/llama.cpp")
        os.system("pip install -q -r /content/llama.cpp/requirements.txt 2>/dev/null || true")

    if not convert_script.exists():
        console.print("[bold red]❌ Failed to find llama.cpp convert script![/bold red]")
        console.print("[yellow]Falling back to manual GGUF export via safetensors...[/yellow]")
        console.print(f"\n[bold]The merged SafeTensors model is saved at:[/bold]")
        console.print(f"  {merged_dir}/")
        console.print(f"\n[bold]To manually convert, run:[/bold]")
        console.print(f"  python /content/llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile {output_dir}/sift-1b-f16.gguf --outtype f16")
        console.print(f"  /content/llama.cpp/llama-quantize {output_dir}/sift-1b-f16.gguf {output_dir}/sift-1b-{quantization}.gguf {quantization}")
        return

    # Step 3a: Convert to f16 GGUF first
    f16_gguf = f"{output_dir}/sift-1b-f16.gguf"
    console.print(f"[yellow]  Step 3a: Converting SafeTensors → F16 GGUF...[/yellow]")
    ret = os.system(f"python {convert_script} {merged_dir} --outfile {f16_gguf} --outtype f16")

    if ret != 0 or not Path(f16_gguf).exists():
        console.print(f"[bold red]❌ F16 conversion failed (exit code {ret})[/bold red]")
        return

    console.print(f"[green]✓ F16 GGUF created: {f16_gguf}[/green]")

    # Step 3b: Quantize to target format
    if quantization != "f16":
        quantized_gguf = f"{output_dir}/sift-1b-{quantization}.gguf"
        console.print(f"[yellow]  Step 3b: Quantizing F16 → {quantization.upper()}...[/yellow]")

        # Build llama-quantize if not already built
        quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_cpp_dir / "llama-quantize"
        if not quantize_bin.exists():
            console.print("[yellow]  Building llama.cpp quantize tool...[/yellow]")
            os.system(f"cd /content/llama.cpp && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)")
            quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

        if quantize_bin.exists():
            ret = os.system(f"{quantize_bin} {f16_gguf} {quantized_gguf} {quantization}")
            if ret == 0 and Path(quantized_gguf).exists():
                console.print(f"[bold green]✓ Quantized GGUF saved: {quantized_gguf}[/bold green]")
                # Remove intermediate f16 to save disk
                Path(f16_gguf).unlink(missing_ok=True)
                final_gguf = quantized_gguf
            else:
                console.print(f"[bold red]❌ Quantization failed. F16 GGUF is still available at {f16_gguf}[/bold red]")
                final_gguf = f16_gguf
        else:
            console.print(f"[yellow]⚠ llama-quantize not found. F16 GGUF available at {f16_gguf}[/yellow]")
            final_gguf = f16_gguf
    else:
        final_gguf = f16_gguf

    # =========================================================================
    # Step 4: Create Ollama Modelfile
    # =========================================================================
    modelfile_path = f"{output_dir}/Modelfile"
    modelfile_content = f"""FROM ./{Path(final_gguf).name}

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.0
PARAMETER num_predict 256
"""
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    console.print(f"[green]✓ Ollama Modelfile created at {modelfile_path}[/green]")

    # =========================================================================
    # Step 5: Push to Hugging Face (optional)
    # =========================================================================
    if push_to_hub:
        console.print(f"\n[yellow]⏳ Pushing GGUF to Hugging Face: {hub_repo}[/yellow]")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True)

        # Upload the GGUF file
        api.upload_file(
            path_or_fileobj=final_gguf,
            path_in_repo=Path(final_gguf).name,
            repo_id=hub_repo,
        )

        # Upload the Modelfile
        api.upload_file(
            path_or_fileobj=modelfile_path,
            path_in_repo="Modelfile",
            repo_id=hub_repo,
        )

        # Upload tokenizer files
        for tok_file in Path(merged_dir).glob("tokenizer*"):
            api.upload_file(
                path_or_fileobj=str(tok_file),
                path_in_repo=tok_file.name,
                repo_id=hub_repo,
            )

        # Upload README
        readme_content = f"""---
license: mit
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - function-calling
  - tool-use
  - gguf
  - sift
---

# Sift-1B GGUF

Quantized GGUF export of **Sift-1B** — a 1.5B-parameter model fine-tuned via SFT + DPO
for deterministic JSON function calling.

## Quick Start

### Ollama
```bash
ollama create sift-1b -f Modelfile
ollama run sift-1b
```

### llama.cpp
```bash
./llama-cli -m sift-1b-{quantization}.gguf -p "<|im_start|>system\\nYou are a function calling agent.\\n<|im_end|>\\n<|im_start|>user\\nWhat is the weather in NYC?<|im_end|>\\n<|im_start|>assistant" --temp 0
```

## Benchmark Results

| Metric | Sift-1B (DPO) | Base Qwen |
|---|---|---|
| Tool Selection Accuracy | **100.0%** | 70.0% |
| Parameter Extraction | **88.0%** | 34.0% |
| JSON Parse Rate | **100.0%** | 96.0% |
| Zero Hallucinations | **100.0%** | 100.0% |

## License

MIT
"""
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=hub_repo,
        )

        console.print(f"[bold green]✓ Pushed to https://huggingface.co/{hub_repo}[/bold green]")

    # =========================================================================
    # Summary
    # =========================================================================
    file_size_mb = Path(final_gguf).stat().st_size / (1024 * 1024) if Path(final_gguf).exists() else 0
    console.print(f"\n[bold green]================================================================[/bold green]")
    console.print(f"[bold green]✓ GGUF Export Complete![/bold green]")
    console.print(f"  File:          {final_gguf}")
    console.print(f"  Size:          {file_size_mb:.1f} MB")
    console.print(f"  Quantization:  {quantization}")
    console.print(f"\n[bold]Test locally with:[/bold]")
    console.print(f"  llama-cli -m {final_gguf} --n-gpu-layers -1")
    console.print(f"\n  Or with Ollama:")
    console.print(f"  ollama create sift-1b -f {output_dir}/Modelfile")
    console.print(f"  ollama run sift-1b")
    console.print(f"[bold green]================================================================[/bold green]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Sift model to GGUF")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path or HF adapter ID")
    parser.add_argument(
        "--quantization", type=str, default="q4_k_m",
        choices=["q4_k_m", "q8_0", "f16", "q5_k_m", "q4_0"],
        help="GGUF quantization method"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument(
        "--hub-repo", type=str, default="SanatanSinghVishen/sift-1b-gguf",
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
