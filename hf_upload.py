from huggingface_hub import HfApi
from pathlib import Path

def hf_upload():
    upload_path = Path(__file__).resolve().parents[0] / "data/matrices/"
    
    api = HfApi()
    
    api.upload_folder(
        folder_path=upload_path,
        repo_id="Carson-Shively/fluency-trainer",
        path_in_repo="matrices/",
        repo_type="dataset"
    )
    
if __name__ == "__main__":
    hf_upload()