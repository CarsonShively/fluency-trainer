from huggingface_hub import HfApi
from pathlib import Path

def hf_upload():
    upload_path = Path(__file__).resolve().parents[0] / "fluency_trainer_model/phone_audio_alignment_model.weights.h5"
    
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=upload_path,
        path_in_repo="phone_audio_alignment_model.weights.h5",
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="model"
    )
    
if __name__ == "__main__":
    hf_upload()