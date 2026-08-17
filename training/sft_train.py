"""
Sift — Supervised Fine-Tuning (SFT) Training Script
=====================================================
Fine-tunes Qwen2.5-1.5B-Instruct using QLoRA (4-bit) with Unsloth
on the curated function-calling SFT dataset.

This is Phase 2 of the Sift pipeline.

Hardware Target: NVIDIA RTX 3050 (4 GB VRAM)
Expected VRAM Usage: ~2.8 GB peak
Expected Training Time: 2-4 hours (10k rows, 3 epochs)

Usage:
  python training/sft_train.py
  python training/sft_train.py --config training/sft_config.yaml
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import json
import yaml
import argparse
from pathlib import Path

from datasets import Dataset
from rich.console import Console

console = Console()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sft_dataset(dataset_path: str) -> Dataset:
    """
    Load the SFT dataset from JSONL and convert to HuggingFace Dataset.
    Each row contains a 'conversations' key with ChatML messages.
    """
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    console.print(f"[green]✓ Loaded {len(rows):,} training rows[/green]")
    return Dataset.from_list(rows)


def formatting_func(examples: dict):
    """
    Format conversations into the tokenizer's chat template.
    Handles both batched dataset mapping and single-item verification passes.
    """
    if "conversations" not in examples:
        return []

    convs = examples["conversations"]
    if not convs:
        return []

    # Case 1: Batched dataset (list of conversations) -> list[list[dict]]
    if isinstance(convs[0], list):
        return [
            tokenizer.apply_chat_template(
                c,
                tokenize=False,
                add_generation_prompt=False,
            )
            for c in convs
        ]
    # Case 2: Single example (list of messages) -> list[dict]
    elif isinstance(convs[0], dict):
        text = tokenizer.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=False,
        )
        return [text]

    return []


def main(config_path: str = None):
    """Main SFT training loop."""
    global tokenizer  # Needed by formatting_func

    if config_path is None:
        config_path = str(Path(__file__).parent / "sft_config.yaml")

    config = load_config(config_path)

    console.print(f"\n[bold cyan]Sift — Supervised Fine-Tuning[/bold cyan]")
    console.print(f"  Model:     {config['model']['name']}")
    console.print(f"  Dataset:   {config['data']['dataset_path']}")
    console.print(f"  Output:    {config['output']['dir']}")
    console.print(f"  Epochs:    {config['training']['num_train_epochs']}")
    console.print(f"  Batch:     {config['training']['per_device_train_batch_size']} "
                  f"(effective: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']})")
    console.print()

    # =========================================================================
    # Step 1: Load model in 4-bit with Unsloth
    # =========================================================================
    console.print("[yellow]⏳ Loading base model in 4-bit...[/yellow]")

    from unsloth import FastLanguageModel

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model"]["name"],
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
            local_files_only=True,
        )
    except Exception:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model"]["name"],
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )

    console.print("[green]✓ Model loaded successfully[/green]")

    # =========================================================================
    # Step 2: Apply LoRA adapters
    # =========================================================================
    console.print("[yellow]⏳ Applying LoRA adapters...[/yellow]")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing=config["lora"]["use_gradient_checkpointing"],
        random_state=config["training"]["seed"],
    )

    # Print trainable parameter stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]✓ LoRA applied — "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)[/green]"
    )

    # =========================================================================
    # Step 3: Load dataset
    # =========================================================================
    dataset = load_sft_dataset(config["data"]["dataset_path"])

    # =========================================================================
    # Step 4: Initialize SFTTrainer
    # =========================================================================
    console.print("[yellow]⏳ Initializing SFT Trainer...[/yellow]")

    from trl import SFTTrainer, SFTConfig

    output_dir = config["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        fp16=config["training"]["fp16"],
        bf16=config["training"].get("bf16", False),
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"].get("save_steps", 250),
        dataset_num_proc=1,
        dataloader_num_workers=0,
        seed=config["training"]["seed"],
        optim=config["training"]["optim"],
        max_seq_length=config["model"]["max_seq_length"],
        dataset_text_field="",           # We use formatting_func instead
        report_to="none",                # Disable W&B / MLflow
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
    )

    # =========================================================================
    # Step 5: Train! (With auto-resume support)
    # =========================================================================
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).exists() else None

    if last_checkpoint:
        console.print(f"[yellow]📦 Resuming seamlessly from {last_checkpoint}...[/yellow]\n")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        console.print("[bold green]🚀 Starting SFT training...[/bold green]\n")
        train_result = trainer.train()

    # Print training summary
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Total steps:  {train_result.global_step}")
    console.print(f"  Final loss:   {train_result.training_loss:.4f}")
    console.print(f"  Runtime:      {train_result.metrics.get('train_runtime', 0):.0f}s")

    # =========================================================================
    # Step 6: Save the fine-tuned model
    # =========================================================================
    console.print(f"\n[yellow]⏳ Saving model to {output_dir}...[/yellow]")

    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    console.print(f"[bold green]✓ Model saved to {output_dir}[/bold green]")
    console.print(f"\n[bold]Next step:[/bold] Run DPO alignment with:")
    console.print(f"  python training/dpo_train.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sift SFT Training")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: training/sft_config.yaml)"
    )
    args = parser.parse_args()
    main(config_path=args.config)
