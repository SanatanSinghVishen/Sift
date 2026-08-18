import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

def main():
    repo_id = "SanatanSinghVishen/sift-1b-sft"
    sft_dir = "checkpoints/sft"
    
    # Check if checkpoint-2500 or latest checkpoint exists
    if Path(f"{sft_dir}/checkpoint-2500").exists():
        upload_path = f"{sft_dir}/checkpoint-2500"
    else:
        upload_path = sft_dir
        
    print(f"Uploading SFT checkpoint from {upload_path} to Hugging Face repo {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=upload_path, repo_id=repo_id)
    print(f"✓ SFT checkpoint uploaded successfully to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
