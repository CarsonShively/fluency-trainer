from huggingface_hub import HfApi
from pathlib import Path

def hf_upload():
    upload_path = Path(__file__).resolve().parents[0] / "evaluation_report.json"
    
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=upload_path,
        path_in_repo="evaluation_report.json",
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="model"
    )
    
if __name__ == "__main__":
    hf_upload()