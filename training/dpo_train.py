"""
Sift — Direct Preference Optimization (DPO) Training Script
=============================================================
Aligns the SFT-trained model to strictly prefer clean JSON outputs
over hallucinated, malformed, or conversational responses.

This is Phase 3 of the Sift pipeline.

The DPO loss function directly optimizes the model's log-probabilities
to increase P(chosen) and decrease P(rejected), without requiring a
separate reward model (unlike PPO/RLHF).

Hardware Target: NVIDIA RTX 3050 (4 GB VRAM)
Expected VRAM Usage: ~3.2 GB peak (DPO processes both chosen+rejected)
Expected Training Time: 1-2 hours (10k pairs, 1 epoch)

Usage:
  python training/dpo_train.py
  python training/dpo_train.py --config training/dpo_config.yaml
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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


def load_dpo_dataset(dataset_path: str) -> Dataset:
    """
    Load the DPO preference dataset from JSONL.
    
    Each row must contain:
      - prompt:   list of messages (system + user)
      - chosen:   list with one assistant message (correct JSON)
      - rejected: list with one assistant message (mutated/broken)
    """
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    console.print(f"[green]✓ Loaded {len(rows):,} preference pairs[/green]")
    return Dataset.from_list(rows)


def main(config_path: str = None):
    """Main DPO training loop."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "dpo_config.yaml")

    config = load_config(config_path)

    console.print(f"\n[bold cyan]Sift — Direct Preference Optimization (DPO)[/bold cyan]")
    console.print(f"  SFT Checkpoint: {config['model']['name']}")
    console.print(f"  Dataset:        {config['data']['dataset_path']}")
    console.print(f"  Output:         {config['output']['dir']}")
    console.print(f"  Beta:           {config['dpo']['beta']}")
    console.print(f"  Learning Rate:  {config['training']['learning_rate']}")
    console.print()

    # =========================================================================
    # Step 1: Patch DPOTrainer with Unsloth optimizations
    # =========================================================================
    from unsloth import FastLanguageModel, PatchDPOTrainer
    PatchDPOTrainer()  # Must be called BEFORE importing DPOTrainer

    from trl import DPOTrainer, DPOConfig

    # =========================================================================
    # Step 2: Load the SFT-trained model
    # =========================================================================
    model_path = config["model"]["name"]
    base_model_id = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

    # If loading a Hugging Face LoRA adapter (e.g. SanatanSinghVishen/sift-1b-sft)
    if "SanatanSinghVishen" in model_path or not Path(model_path).exists():
        console.print(f"[yellow]⏳ Loading base model {base_model_id}...[/yellow]")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )
        console.print(f"[yellow]⏳ Attaching SFT adapter from {model_path}...[/yellow]")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )

    console.print("[green]✓ SFT model loaded[/green]")

    # =========================================================================
    # Step 3: Re-apply LoRA for DPO phase
    # =========================================================================
    console.print("[yellow]⏳ Applying LoRA adapters for DPO...[/yellow]")

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]✓ LoRA applied — "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)[/green]"
    )

    # =========================================================================
    # Step 4: Load DPO dataset
    # =========================================================================
    dataset = load_dpo_dataset(config["data"]["dataset_path"])

    # =========================================================================
    # Step 5: Initialize DPOTrainer
    # =========================================================================
    console.print("[yellow]⏳ Initializing DPO Trainer...[/yellow]")

    output_dir = config["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        fp16=config["training"]["fp16"],
        bf16=config["training"].get("bf16", True),
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"].get("save_steps", 500),
        dataset_num_proc=1,
        dataloader_num_workers=0,
        seed=config["training"]["seed"],
        optim=config["training"]["optim"],
        max_length=config["model"]["max_seq_length"],
        max_prompt_length=config["model"]["max_seq_length"] // 2,
        beta=config["dpo"]["beta"],
        loss_type=config["dpo"]["loss_type"],
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # =========================================================================
    # Step 6: Train! (With auto-resume support)
    # =========================================================================
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).exists() else None

    if last_checkpoint:
        console.print(f"[yellow]📦 Resuming DPO from {last_checkpoint}...[/yellow]\n")
        train_result = dpo_trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        console.print("[bold green]🚀 Starting DPO alignment...[/bold green]\n")
        train_result = dpo_trainer.train()

    console.print(f"\n[bold green]✓ DPO alignment complete![/bold green]")
    console.print(f"  Total steps:  {train_result.global_step}")
    console.print(f"  Final loss:   {train_result.training_loss:.4f}")
    console.print(f"  Runtime:      {train_result.metrics.get('train_runtime', 0):.0f}s")

    # =========================================================================
    # Step 7: Save the aligned model
    # =========================================================================
    console.print(f"\n[yellow]⏳ Saving aligned model to {output_dir}...[/yellow]")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    console.print(f"[bold green]✓ Model saved to {output_dir}[/bold green]")
    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"  1. Evaluate:  python eval/evaluate.py")
    console.print(f"  2. Export:    python export/export_gguf.py")
    console.print(f"  3. Deploy:    python export/push_to_hub.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sift DPO Training")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: training/dpo_config.yaml)"
    )
    args = parser.parse_args()
    main(config_path=args.config)
