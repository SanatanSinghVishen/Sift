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

    from trl import DPOTrainer
    from transformers import Trainer, TrainingArguments

    # Universal fix for Trainer._run_epoch calling get_batch_samples with varying arguments
    def safe_get_batch_samples(self, epoch_iterator, num_batches, device=None):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        return batch_samples, num_items_in_batch

    DPOTrainer.get_batch_samples = safe_get_batch_samples
    Trainer.get_batch_samples = safe_get_batch_samples

    # Universal fix for Trainer.training_step passing num_items_in_batch to compute_loss in transformers >= 4.46
    _orig_compute_loss = DPOTrainer.compute_loss
    def safe_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)

    DPOTrainer.compute_loss = safe_compute_loss

    # DPOConfig exists in trl >= 0.9.0; older versions use TrainingArguments
    try:
        from trl import DPOConfig
        _has_dpo_config = True
    except ImportError:
        _has_dpo_config = False
        console.print("[dim]ℹ Using TrainingArguments (trl < 0.9.0 detected)[/dim]")

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
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )

    # Ensure pad token is defined for DPO data collation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Safe DPO data collator to handle None values in attention masks or labels
    import torch
    from torch.nn.utils.rnn import pad_sequence
    import trl.trainer.utils as trl_utils
    import trl.trainer.dpo_trainer as dpo_module

    class SafeDPODataCollatorWithPadding:
        def __init__(self, pad_token_id=151643, label_pad_token_id=-100, is_encoder_decoder=False):
            self.pad_token_id = pad_token_id if pad_token_id is not None else 151643
            self.label_pad_token_id = label_pad_token_id if label_pad_token_id is not None else -100
            self.is_encoder_decoder = is_encoder_decoder

        def __call__(self, features):
            padded_batch = {}
            for k in features[0].keys():
                if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                    to_pad = []
                    for ex in features:
                        val = ex.get(k)
                        if val is None:
                            if k.endswith("_attention_mask"):
                                match_k = k.replace("_attention_mask", "_input_ids")
                                ids_len = len(ex.get(match_k) or [])
                                val = [1] * ids_len
                            elif k.endswith("_labels"):
                                match_k = k.replace("_labels", "_input_ids")
                                ids_len = len(ex.get(match_k) or [])
                                val = [self.label_pad_token_id] * ids_len
                            else:
                                val = [self.pad_token_id]
                        
                        if isinstance(val, torch.Tensor):
                            val = val.tolist()
                        elif not isinstance(val, list):
                            val = [val] if val is not None else [0]
                        
                        clean_val = [x if x is not None else (0 if "attention_mask" in k else self.pad_token_id) for x in val]
                        if len(clean_val) == 0:
                            clean_val = [0 if "attention_mask" in k else self.pad_token_id]

                        if "prompt" in k:
                            to_pad.append(torch.tensor(clean_val[::-1], dtype=torch.long))
                        else:
                            to_pad.append(torch.tensor(clean_val, dtype=torch.long))

                    if k.endswith("_input_ids"):
                        pad_val = self.pad_token_id
                    elif k.endswith("_labels"):
                        pad_val = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        pad_val = 0
                    else:
                        pad_val = self.pad_token_id

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=pad_val)
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
                elif isinstance(features[0][k], (int, float)):
                    padded_batch[k] = torch.tensor([ex[k] for ex in features])
            return padded_batch

    trl_utils.DPODataCollatorWithPadding = SafeDPODataCollatorWithPadding
    dpo_module.DPODataCollatorWithPadding = SafeDPODataCollatorWithPadding
    collator = SafeDPODataCollatorWithPadding(pad_token_id=tokenizer.pad_token_id)

    console.print("[green]✓ SFT model loaded[/green]")

    # =========================================================================
    # Step 3: Re-apply / Enable LoRA for DPO phase
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
            use_gradient_checkpointing=config["lora"]["use_gradient_checkpointing"],
            random_state=config["training"]["seed"],
        )
    else:
        FastLanguageModel.for_training(model)
        for name, param in model.named_parameters():
            if "lora" in name.lower() or "adapter" in name.lower():
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]✓ LoRA active — "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)[/green]"
    )

    # =========================================================================
    # Step 4: Load DPO dataset
    # =========================================================================
    dataset = load_dpo_dataset(config["data"]["dataset_path"])

    # ---- Format dataset for trl compatibility --------------------------------
    # trl < 0.9.0 expects prompt/chosen/rejected as plain strings.
    # Our dataset stores them as ChatML message lists. Apply chat_template
    # to flatten them when needed.
    if not _has_dpo_config:
        console.print("[dim]ℹ Formatting dataset for trl < 0.9.0 (messages → strings)...[/dim]")

        def _format_row(row):
            # prompt: list of {role, content} → single string via chat template
            prompt_str = tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True
            )
            # chosen/rejected: list with one assistant message → extract content
            chosen_str = row["chosen"][0]["content"] if isinstance(row["chosen"], list) else row["chosen"]
            rejected_str = row["rejected"][0]["content"] if isinstance(row["rejected"], list) else row["rejected"]
            return {"prompt": prompt_str, "chosen": chosen_str, "rejected": rejected_str}

        dataset = dataset.map(_format_row, num_proc=1)
        console.print(f"[green]✓ Dataset formatted ({len(dataset):,} rows)[/green]")

    # =========================================================================
    # Step 5: Initialize DPOTrainer
    # =========================================================================
    console.print("[yellow]⏳ Initializing DPO Trainer...[/yellow]")

    output_dir = config["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Auto-detect precision (bf16 may fail on T4 / older drivers) --------
    import torch
    import math
    _use_bf16 = False
    _use_fp16 = False
    if torch.cuda.is_available():
        try:
            _use_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            _use_bf16 = False
        _use_fp16 = not _use_bf16
    console.print(f"[dim]ℹ Precision: {'bf16' if _use_bf16 else 'fp16'}[/dim]")

    # ---- Convert warmup_ratio → warmup_steps (deprecated in transformers v5) -
    _batch = config["training"]["per_device_train_batch_size"]
    _accum = config["training"]["gradient_accumulation_steps"]
    _epochs = config["training"]["num_train_epochs"]
    _total_steps = math.ceil(len(dataset) / (_batch * _accum)) * _epochs
    _warmup_steps = int(config["training"]["warmup_ratio"] * _total_steps)
    console.print(f"[dim]ℹ Warmup: {_warmup_steps} steps ({config['training']['warmup_ratio']} × {_total_steps})[/dim]")

    if _has_dpo_config:
        # trl >= 0.9.0: all args go into DPOConfig
        training_args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=_batch,
            gradient_accumulation_steps=_accum,
            num_train_epochs=_epochs,
            learning_rate=config["training"]["learning_rate"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            warmup_steps=_warmup_steps,
            weight_decay=config["training"]["weight_decay"],
            fp16=_use_fp16,
            bf16=_use_bf16,
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
            remove_unused_columns=False,
            report_to="none",
        )
        dpo_trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
    else:
        # trl < 0.9.0: DPO-specific args go into DPOTrainer constructor
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=_batch,
            gradient_accumulation_steps=_accum,
            num_train_epochs=_epochs,
            learning_rate=config["training"]["learning_rate"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            warmup_steps=_warmup_steps,
            weight_decay=config["training"]["weight_decay"],
            fp16=_use_fp16,
            bf16=_use_bf16,
            logging_steps=config["training"]["logging_steps"],
            save_strategy=config["training"]["save_strategy"],
            save_steps=config["training"].get("save_steps", 500),
            seed=config["training"]["seed"],
            optim=config["training"]["optim"],
            remove_unused_columns=False,
            report_to="none",
        )

        # trl<0.9 passes `tokenizer=` to Trainer.__init__(), but
        # transformers v5 renamed it to `processing_class`. Monkey-patch
        # Trainer.__init__ to accept both names seamlessly.
        import inspect
        from transformers import Trainer as _Trainer
        _orig_init = _Trainer.__init__
        _trainer_params = inspect.signature(_orig_init).parameters
        if "tokenizer" not in _trainer_params and "processing_class" in _trainer_params:
            def _patched_init(self, *args, **kwargs):
                if "tokenizer" in kwargs:
                    kwargs["processing_class"] = kwargs.pop("tokenizer")
                return _orig_init(self, *args, **kwargs)
            _Trainer.__init__ = _patched_init
            console.print("[dim]ℹ Patched Trainer.__init__ for tokenizer→processing_class[/dim]")

        dpo_trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            beta=config["dpo"]["beta"],
            loss_type=config["dpo"]["loss_type"],
            max_length=config["model"]["max_seq_length"],
            max_prompt_length=config["model"]["max_seq_length"] // 2,
        )

    # Bind safe collator, safe_get_batch_samples, and safe_compute_loss directly to the trainer instance
    dpo_trainer.data_collator = collator
    dpo_trainer.get_batch_samples = safe_get_batch_samples.__get__(dpo_trainer, type(dpo_trainer))
    dpo_trainer.compute_loss = safe_compute_loss.__get__(dpo_trainer, type(dpo_trainer))

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
