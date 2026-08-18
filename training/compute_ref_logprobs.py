"""
Sift — Precompute & Save Reference Log-Probabilities
=====================================================
Computes reference log-probabilities once and saves them permanently
to both local disk (checkpoints/dpo/ref_logprobs.pt) and Google Drive
(/content/drive/MyDrive/sift_dpo_backup/ref_logprobs.pt).

Usage:
  python training/compute_ref_logprobs.py
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import json
import yaml
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from rich.console import Console

console = Console()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dpo_dataset(dataset_path: str) -> Dataset:
    rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    console.print(f"[green]✓ Loaded {len(rows):,} preference pairs[/green]")
    return Dataset.from_list(rows)


def compute_log_probs(logits, labels, mask):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = torch.clamp(shift_labels, min=0)
    per_token_log_probs = log_probs.gather(2, safe_labels.unsqueeze(2)).squeeze(2)
    per_token_log_probs = per_token_log_probs * shift_mask
    return per_token_log_probs.sum(dim=-1)


def main(config_path: str = None):
    if config_path is None:
        config_path = str(Path(__file__).parent / "dpo_config.yaml")

    config = load_config(config_path)

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    local_cache_path = output_dir / "ref_logprobs.pt"
    drive_backup_dir = Path("/content/drive/MyDrive/sift_dpo_backup")

    # If already cached, check if we need to copy to Drive
    if local_cache_path.exists():
        console.print(f"[bold green]✓ Reference cache already exists at {local_cache_path}![/bold green]")
        if Path("/content/drive/MyDrive").exists():
            drive_backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_cache_path, drive_backup_dir / "ref_logprobs.pt")
            console.print(f"[bold green]✓ Copied to Google Drive: {drive_backup_dir / 'ref_logprobs.pt'}[/bold green]")
        return

    console.print(f"\n[bold cyan]Sift — Precomputing Reference Log-Probabilities[/bold cyan]")
    console.print(f"  SFT Checkpoint: {config['model']['name']}")
    console.print(f"  Dataset:        {config['data']['dataset_path']}")
    console.print(f"  Output:         {local_cache_path}\n")

    # Step 1: Load base model + SFT adapter
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
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Step 2: Tokenize pairs
    raw_dataset = load_dpo_dataset(config["data"]["dataset_path"])
    max_len = config["model"]["max_seq_length"]
    max_prompt_len = max_len // 2

    console.print("[yellow]⏳ Tokenizing dataset...[/yellow]")

    def tokenize_pair(row):
        prompt_msgs = row["prompt"]
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        chosen_text = row["chosen"][0]["content"] if isinstance(row["chosen"], list) else row["chosen"]
        rejected_text = row["rejected"][0]["content"] if isinstance(row["rejected"], list) else row["rejected"]

        prompt_enc = tokenizer(prompt_str, truncation=True, max_length=max_prompt_len, add_special_tokens=False)
        chosen_enc = tokenizer(prompt_str + chosen_text + tokenizer.eos_token, truncation=True, max_length=max_len, add_special_tokens=False)
        rejected_enc = tokenizer(prompt_str + rejected_text + tokenizer.eos_token, truncation=True, max_length=max_len, add_special_tokens=False)

        prompt_len = len(prompt_enc["input_ids"])
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels": [-100] * min(prompt_len, len(chosen_enc["input_ids"])) + chosen_enc["input_ids"][prompt_len:],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels": [-100] * min(prompt_len, len(rejected_enc["input_ids"])) + rejected_enc["input_ids"][prompt_len:],
        }

    tokenized = raw_dataset.map(tokenize_pair, remove_columns=raw_dataset.column_names, num_proc=1)
    console.print(f"[green]✓ Tokenized {len(tokenized):,} pairs[/green]")

    # Step 3: Dataloader with batch size 16 (Fast ~5 min)
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        result = {}
        for key in ["chosen_input_ids", "chosen_attention_mask", "chosen_labels",
                     "rejected_input_ids", "rejected_attention_mask", "rejected_labels"]:
            seqs = [torch.tensor(ex[key], dtype=torch.long) for ex in batch]
            pad_val = -100 if "labels" in key else (0 if "mask" in key else pad_id)
            max_seq_len = max(s.size(0) for s in seqs)
            result[key] = torch.stack([
                torch.cat([torch.full((max_seq_len - s.size(0),), pad_val, dtype=torch.long), s])
                for s in seqs
            ])
        return result

    ref_dataloader = DataLoader(
        tokenized,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Step 4: Compute Reference Pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _use_bf16 = False
    if torch.cuda.is_available():
        try:
            _use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            _use_bf16 = False
    amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16

    model.eval()
    ref_chosen_logps_all = []
    ref_rejected_logps_all = []

    console.print(f"[yellow]🚀 Computing reference log-probabilities with batch_size=16 ({len(ref_dataloader)} batches)...[/yellow]")

    with torch.no_grad():
        for i, batch in enumerate(ref_dataloader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

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

            if (i + 1) % 50 == 0 or (i + 1) == len(ref_dataloader):
                console.print(f"  [dim]Progress: {i + 1}/{len(ref_dataloader)} batches ({100 * (i + 1) / len(ref_dataloader):.1f}%)[/dim]")

    ref_chosen_logps_all = torch.cat(ref_chosen_logps_all)
    ref_rejected_logps_all = torch.cat(ref_rejected_logps_all)

    # Save to local disk
    torch.save({"chosen": ref_chosen_logps_all, "rejected": ref_rejected_logps_all}, local_cache_path)
    console.print(f"\n[bold green]✓ Reference log-probs permanently saved to {local_cache_path}![/bold green]")

    # Save to Google Drive if mounted
    if Path("/content/drive/MyDrive").exists():
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_cache_path, drive_backup_dir / "ref_logprobs.pt")
        console.print(f"[bold green]✓ Mirrored to Google Drive: {drive_backup_dir / 'ref_logprobs.pt'}[/bold green]")

    console.print("\n[bold]Now you can run training with: python training/dpo_train.py (loads cache in 1 sec!)[/bold]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Reference Log-Probs")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
