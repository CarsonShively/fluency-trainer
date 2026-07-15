from transformers import AutoProcessor, AutoModelForCTC
import torch
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import shutil
from huggingface_hub import snapshot_download, HfApi, get_token


def extract_user_audio_logits():
    print("Starting...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU available")
    
    device = torch.device("cuda")
    print("device created sucessfully.")
    
    local_data = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns=["samples/**", "speechocean672/**"]
    ))
    
    out_path = Path(__file__).resolve().parents[0] / "user_audio"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
        
    out_path.mkdir(parents=True, exist_ok=True)
      
    train_out_path = out_path / "train"
    train_out_path.mkdir(parents=True, exist_ok=True)
    
    val_out_path = out_path / "val"
    val_out_path.mkdir(parents=True, exist_ok=True)
    
    test_out_path = out_path / "test"
    test_out_path.mkdir(parents=True, exist_ok=True)
    
    expected_sample_rate = 16000
    
    
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model.eval()
    model.to(device)
    
    
    
    with open(local_data / "samples/train.json", "r") as con:
        train_samples = json.load(con)
        
    with open(local_data / "samples/val.json", "r") as con:
        val_samples = json.load(con)
        
    with open(local_data / "samples/test.json", "r") as con:
        test_samples = json.load(con)
    
    train_samples_counter = len(train_samples)
    for sample_id, sample in train_samples.items():
        
        audio_path = local_data / sample["user_audio_path"]
        waveform, sample_rate = sf.read(audio_path)
            
        if sample_rate != expected_sample_rate:
            raise ValueError(f"sample rate: {sample_rate}, expected sample rate: {expected_sample_rate}")
            
        processed_audio = processor(
                waveform,
                sampling_rate=expected_sample_rate,
                return_tensors="pt"
            ).to(device)
        
        
        with torch.inference_mode():
            output_audio = model(**processed_audio) 
            
        output_logits = output_audio.logits
        output_numpy = output_logits.squeeze(0).cpu().numpy()
        
        np.save(train_out_path / f"{sample_id}.npy", output_numpy)
        train_samples_counter -= 1
        print(f"train samples remaining: {train_samples_counter}")
        
    val_samples_counter = len(val_samples)
    for sample_id, sample in val_samples.items():
        
        audio_path = local_data / sample["user_audio_path"]
        waveform, sample_rate = sf.read(audio_path)
            
        if sample_rate != expected_sample_rate:
            raise ValueError(f"sample rate: {sample_rate}, expected sample rate: {expected_sample_rate}")
            
        processed_audio = processor(
                waveform,
                sampling_rate=expected_sample_rate,
                return_tensors="pt"
            ).to(device)
        
        with torch.inference_mode():
            output_audio = model(**processed_audio) 
            
        output_logits = output_audio.logits
        output_numpy = output_logits.squeeze(0).cpu().numpy()
        
        np.save(val_out_path/ f"{sample_id}.npy", output_numpy)
        val_samples_counter -= 1
        print(f"val samples remaining: {val_samples_counter}")
        
        
    test_samples_counter = len(test_samples)
    for sample_id, sample in test_samples.items():
        
        audio_path = local_data / sample["user_audio_path"]
        waveform, sample_rate = sf.read(audio_path)
            
        if sample_rate != expected_sample_rate:
            raise ValueError(f"sample rate: {sample_rate}, expected sample rate: {expected_sample_rate}")
            
        processed_audio = processor(
                waveform,
                sampling_rate=expected_sample_rate,
                return_tensors="pt"
            ).to(device)
        
        with torch.inference_mode():
            output_audio = model(**processed_audio) 
            
        output_logits = output_audio.logits
        output_numpy = output_logits.squeeze(0).cpu().numpy()
        
        np.save(test_out_path / f"{sample_id}.npy", output_numpy)
        test_samples_counter -= 1
        print(f"test samples remaining: {test_samples_counter}")
        
    print("user audio complete")
    
    if get_token() != None:
        api = HfApi()
        
        api.upload_folder(
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="dataset",
            delete_patterns="user_audio/**",
            path_in_repo="user_audio",
            upload_path=out_path
        )

    
if __name__ == "__main__":
    extract_user_audio_logits()