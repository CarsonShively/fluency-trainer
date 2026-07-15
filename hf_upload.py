from huggingface_hub import HfApi
from pathlib import Path

def hf_upload():
    upload_path = Path(__file__).resolve().parents[0] / "sentences"
    
    api = HfApi()
    
    api.upload_folder(
        folder_path=upload_path,
        path_in_repo="sentences",
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset"
    )
    
if __name__ == "__main__":
    hf_upload()