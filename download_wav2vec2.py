from pathlib import Path
from transformers import AutoProcessor, AutoModelForCTC
import shutil

def download_wav2vec2():
 
    
    
    download_url = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    wav2vec2_path = Path(__file__).resolve().parents[0] / "wav2vec2"
    if wav2vec2_path.is_dir():
        shutil.rmtree(wav2vec2_path)
    wav2vec2_path.mkdir(parents=True, exist_ok=True)
    
    processor = AutoProcessor.from_pretrained(download_url)
    model = AutoModelForCTC.from_pretrained(download_url)
    
    processor.save_pretrained(wav2vec2_path)
    model.save_pretrained(wav2vec2_path)
    
if __name__ == "__main__":
    download_wav2vec2()