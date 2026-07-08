import soundfile as sf
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCTC
import torch
import json
import numpy as np
import shutil

def user_audio_matrix():

    print("starting...")
    samples_path = Path(__file__).resolve().parents[0] / "data/samples"
    
    with open(samples_path / "train.json", "r") as con:
        train_samples = json.load(con)
    
    with open(samples_path / "val.json", "r") as con:
        val_samples = json.load(con)
    
    with open(samples_path / "test.json", "r") as con:
        test_samples = json.load(con)
    
    print("train, val, and test samples loaded")
    
    expected_sample_rate = 16000
    
    train_user_audio = []
    val_user_audio = []
    test_user_audio = []
    
    splits = ((train_samples, train_user_audio, "train"), (val_samples, val_user_audio, "val"), (test_samples, test_user_audio, "test"))
    
    out_path = Path(__file__).resolve().parents[0] / "data/matrices/user_audio"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    for samples, split, out in splits:
      
        for audio_path in samples.values():
            waveform, sample_rate = sf.read(audio_path["user_audio_path"], dtype="float32")
            
            if sample_rate != expected_sample_rate:
                raise ValueError("sample rate not 16000, wav2vec2 expects 16000")
            
            split.append(waveform)
        wav2vec2_path = Path(__file__).resolve().parents[0] / "wav2vec2"
        
        processor = AutoProcessor.from_pretrained(wav2vec2_path, local_files_only=True)
        model = AutoModelForCTC.from_pretrained(wav2vec2_path, local_files_only=True)
        
        processed_user_audio = processor(
            train_user_audio,
            sample_rate=expected_sample_rate,
            return_tensor="pt",
            padding=True
        )
        
        with torch.no_grad():
            model.eval()
            output = model(**processed_user_audio)
            
        output_logits = output.logits
        output_softmax = torch.softmax(output_logits, axis=-1)
    
        output_audio = output_softmax.cpu().tonumpy()
        output_mask = processed_user_audio["attention_mask"].cpu().tonumpy()
    
        np.save(out_path / f"{out}.npy", output_audio)
        np.save(out_path / f"{out}_mask.npy", output_mask)
        
if __name__ == "__main__":
    user_audio_matrix()