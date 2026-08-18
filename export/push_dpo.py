import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Push DPO adapter to Hugging Face Hub")
    parser.add_argument(
        "--source", type=str,
        default="checkpoints/dpo",
        help="Path to checkpoint directory (e.g. /content/drive/MyDrive/sift_dpo_backup/checkpoint-500)",
    )
    parser.add_argument(
        "--repo-id", type=str,
        default="SanatanSinghVishen/sift-1b-dpo",
        help="Hugging Face repo ID",
    )
    args = parser.parse_args()

    dpo_dir = args.source
    if not Path(dpo_dir).exists():
        print(f"Error: {dpo_dir} not found!")
        sys.exit(1)

    print(f"Uploading optimal DPO model from {dpo_dir} to Hugging Face repo {args.repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=dpo_dir,
        repo_id=args.repo_id,
        ignore_patterns=["training_state.pt", "ref_logprobs.pt"],
    )
    print(f"\n✓ Optimal DPO model uploaded successfully to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
