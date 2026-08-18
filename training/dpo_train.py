"""
Sift — Direct Preference Optimization (DPO) Training Script
=============================================================
Aligns the SFT-trained model to strictly prefer clean JSON outputs
over hallucinated, malformed, or conversational responses.

This is Phase 3 of the Sift pipeline.

This script implements DPO from scratch using only PyTorch + transformers,
avoiding all trl version-compatibility issues.

Hardware Target: NVIDIA T4 / RTX 3050 (4–16 GB VRAM)
Expected Training Time: 1-2 hours (10k pairs, 1 epoch)

Usage:
  python training/dpo_train.py
  python training/dpo_train.py --config training/dpo_config.yaml
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import json
import yaml
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
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
    path = Path(dataset_path)
    if not path.exists():
        console.print(f"[yellow]⚠️ Dataset {dataset_path} not found. Auto-generating...[/yellow]")
        sft_path = Path("data/sft_dataset.jsonl")
        import subprocess
        import sys
        if not sft_path.exists():
            console.print("[yellow]⏳ Generating SFT dataset first via prepare_sft.py...[/yellow]")
            subprocess.run([sys.executable, "data/prepare_sft.py"], check=True)
        console.print("[yellow]⏳ Generating DPO dataset via generate_dpo.py...[/yellow]")
        subprocess.run([sys.executable, "data/generate_dpo.py"], check=True)

    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    console.print(f"[green]✓ Loaded {len(rows):,} preference pairs[/green]")
    return Dataset.from_list(rows)


def compute_log_probs(logits, labels, mask):
    """Compute per-token log probabilities for the given labels."""
    # logits: (B, T, V), labels: (B, T), mask: (B, T)
    # Shift so that token n predicts token n+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    # Per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # Mask out padding and sum
    per_token_log_probs = per_token_log_probs * shift_mask
    return per_token_log_probs.sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1):
    """Compute the DPO loss (sigmoid variant)."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    # Metrics
    chosen_rewards = (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = (policy_rejected_logps - ref_rejected_logps).detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
    return loss, reward_margin


def main(config_path: str = None):
    """Main DPO training loop — implemented from scratch, no trl dependency."""
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
    # Step 1: Load model with Unsloth
    # =========================================================================
    from unsloth import FastLanguageModel

    model_path = config["model"]["name"]
    base_model_id = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

    if "SanatanSinghVishen" in model_path or not Path(model_path).exists():
        console.print(f"[yellow]⏳ Loading base model {base_model_id}...[/yellow]")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )
        console.print(f"[yellow]⏳ Attaching SFT adapter from {model_path}...[/yellow]")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    console.print("[green]✓ SFT model loaded[/green]")

    # =========================================================================
    # Step 2: Configure LoRA for DPO
    # =========================================================================
    console.print("[yellow]⏳ Configuring LoRA adapters for DPO...[/yellow]")

    if not hasattr(model, "peft_config"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["lora_dropout"],
            bias=config["lora"]["bias"],
            use_gradient_checkpointing=False,  # Disabled for transformers v5 compatibility
            random_state=config["training"]["seed"],
        )
    else:
        # Already has LoRA — just enable training
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "adapter" in name.lower():
                param.requires_grad = True

    # Explicitly disable gradient checkpointing to avoid _gradient_checkpointing_func errors
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]✓ LoRA active — "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)[/green]"
    )

    # =========================================================================
    # Step 3: Load & tokenize DPO dataset
    # =========================================================================
    raw_dataset = load_dpo_dataset(config["data"]["dataset_path"])

    max_len = config["model"]["max_seq_length"]
    max_prompt_len = max_len // 2

    console.print("[yellow]⏳ Tokenizing preference pairs...[/yellow]")

    def tokenize_pair(row):
        """Tokenize a single DPO preference pair into prompt + chosen/rejected."""
        # Build prompt string
        prompt_msgs = row["prompt"]
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        # Extract chosen/rejected text
        chosen_text = row["chosen"][0]["content"] if isinstance(row["chosen"], list) else row["chosen"]
        rejected_text = row["rejected"][0]["content"] if isinstance(row["rejected"], list) else row["rejected"]

        # Tokenize prompt
        prompt_enc = tokenizer(prompt_str, truncation=True, max_length=max_prompt_len,
                               add_special_tokens=False)

        # Tokenize chosen response (prompt + chosen)
        chosen_full = prompt_str + chosen_text + tokenizer.eos_token
        chosen_enc = tokenizer(chosen_full, truncation=True, max_length=max_len,
                               add_special_tokens=False)

        # Tokenize rejected response (prompt + rejected)
        rejected_full = prompt_str + rejected_text + tokenizer.eos_token
        rejected_enc = tokenizer(rejected_full, truncation=True, max_length=max_len,
                                 add_special_tokens=False)

        prompt_len = len(prompt_enc["input_ids"])

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels": [-100] * min(prompt_len, len(chosen_enc["input_ids"])) +
                             chosen_enc["input_ids"][prompt_len:],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels": [-100] * min(prompt_len, len(rejected_enc["input_ids"])) +
                               rejected_enc["input_ids"][prompt_len:],
        }

    tokenized = raw_dataset.map(tokenize_pair, remove_columns=raw_dataset.column_names, num_proc=1)
    console.print(f"[green]✓ Tokenized {len(tokenized):,} pairs[/green]")

    # =========================================================================
    # Step 4: Collator & DataLoader
    # =========================================================================
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        """Pad chosen and rejected sequences to the same length within the batch."""
        result = {}
        for key in ["chosen_input_ids", "chosen_attention_mask", "chosen_labels",
                     "rejected_input_ids", "rejected_attention_mask", "rejected_labels"]:
            seqs = [torch.tensor(ex[key], dtype=torch.long) for ex in batch]
            pad_val = -100 if "labels" in key else (0 if "mask" in key else pad_id)
            max_seq_len = max(s.size(0) for s in seqs)
            padded = []
            for s in seqs:
                padding = torch.full((max_seq_len - s.size(0),), pad_val, dtype=torch.long)
                padded.append(torch.cat([padding, s]) if "chosen" in key or "rejected" in key else torch.cat([s, padding]))
                # Left-pad for causal LM
            result[key] = torch.stack([torch.cat([torch.full((max_seq_len - s.size(0),), pad_val, dtype=torch.long), s]) for s in seqs])
        return result

    batch_size = config["training"]["per_device_train_batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # =========================================================================
    # Step 5: Compute reference log-probs (frozen model)
    # =========================================================================
    console.print("[yellow]⏳ Computing reference log-probabilities (frozen pass)...[/yellow]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use fp16 or bf16 based on hardware
    _use_bf16 = False
    if torch.cuda.is_available():
        try:
            _use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            _use_bf16 = False
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16
    console.print(f"[dim]ℹ Precision: {'bf16' if _use_bf16 else 'fp16'}[/dim]")

    model.eval()
    ref_chosen_logps_all = []
    ref_rejected_logps_all = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # Create response mask (where labels != -100)
            chosen_resp_mask = (chosen_labels != -100).long()
            rejected_resp_mask = (rejected_labels != -100).long()

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                chosen_out = model(input_ids=chosen_ids, attention_mask=chosen_mask)
                rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask)

            ref_chosen_logps_all.append(
                compute_log_probs(chosen_out.logits.float(), chosen_ids, chosen_resp_mask).cpu()
            )
            ref_rejected_logps_all.append(
                compute_log_probs(rejected_out.logits.float(), rejected_ids, rejected_resp_mask).cpu()
            )

            if (i + 1) % 500 == 0:
                console.print(f"[dim]  Reference pass: {i + 1}/{len(dataloader)} batches[/dim]")

    ref_chosen_logps_all = torch.cat(ref_chosen_logps_all)
    ref_rejected_logps_all = torch.cat(ref_rejected_logps_all)
    console.print(f"[green]✓ Reference log-probs computed ({len(ref_chosen_logps_all):,} pairs)[/green]")

    # =========================================================================
    # Step 6: DPO Training Loop
    # =========================================================================
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999),
    )

    total_steps = math.ceil(len(dataloader) / grad_accum)
    warmup_steps = int(config["training"]["warmup_ratio"] * total_steps)

    # Cosine LR scheduler with warmup
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    console.print(f"\n[bold green]🚀 Starting DPO alignment...[/bold green]")
    console.print(f"  Total steps:    {total_steps}")
    console.print(f"  Warmup steps:   {warmup_steps}")
    console.print(f"  Batch size:     {batch_size} × {grad_accum} = {batch_size * grad_accum}")
    console.print(f"  Beta:           {config['dpo']['beta']}\n")

    output_dir = config["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    beta = config["dpo"]["beta"]
    log_steps = config["training"]["logging_steps"]
    save_steps = config["training"].get("save_steps", 500)
    global_step = 0
    running_loss = 0.0
    running_margin = 0.0
    sample_idx = 0

    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad()

    for batch_i, batch in enumerate(dataloader):
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        chosen_resp_mask = (chosen_labels != -100).long()
        rejected_resp_mask = (rejected_labels != -100).long()

        # Get reference log-probs for this batch
        bs = chosen_ids.size(0)
        ref_c = ref_chosen_logps_all[sample_idx:sample_idx + bs].to(device)
        ref_r = ref_rejected_logps_all[sample_idx:sample_idx + bs].to(device)
        sample_idx += bs

        # Forward pass
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            chosen_out = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_out = model(input_ids=rejected_ids, attention_mask=rejected_mask)

            policy_chosen_logps = compute_log_probs(
                chosen_out.logits.float(), chosen_ids, chosen_resp_mask
            )
            policy_rejected_logps = compute_log_probs(
                rejected_out.logits.float(), rejected_ids, rejected_resp_mask
            )

            loss, reward_margin = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_c, ref_r, beta=beta
            )
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        running_loss += loss.item()
        running_margin += reward_margin

        if (batch_i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_steps == 0:
                avg_loss = running_loss / log_steps
                avg_margin = running_margin / (log_steps * grad_accum)
                lr = scheduler.get_last_lr()[0]
                console.print(
                    f"  Step {global_step:>5}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Reward Margin: {avg_margin:+.3f} | "
                    f"LR: {lr:.2e}"
                )
                running_loss = 0.0
                running_margin = 0.0

            if global_step % save_steps == 0:
                ckpt_dir = f"{output_dir}/checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                console.print(f"  [dim]💾 Saved checkpoint at step {global_step}[/dim]")

    # =========================================================================
    # Step 7: Save final model
    # =========================================================================
    console.print(f"\n[bold green]✓ DPO alignment complete![/bold green]")
    console.print(f"  Total steps:    {global_step}")

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
