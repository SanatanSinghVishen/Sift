import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

def main():
    repo_id = "SanatanSinghVishen/sift-1b-dpo"
    dpo_dir = "checkpoints/dpo"
    
    # Check if final checkpoints/dpo exists, or if latest checkpoint directory
    if not Path(dpo_dir).exists():
        print(f"Error: {dpo_dir} not found!")
        sys.exit(1)
        
    print(f"Uploading DPO aligned model from {dpo_dir} to Hugging Face repo {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=dpo_dir,
        repo_id=repo_id,
        ignore_patterns=["ref_logprobs.pt", "checkpoint-*/*"], # Ignore large intermediate states/cache
    )
    print(f"\n✓ DPO aligned model uploaded successfully to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
