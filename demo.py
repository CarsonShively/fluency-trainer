from huggingface_hub import snapshot_download
from pathlib import Path

def dmeo():
    local_data = Path(__file__).resolve().parents[0] / "data"
    
    snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        local_dir=local_data,
        allow_patterns=["sentences/**"]
    )
    
    