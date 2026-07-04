import soundfile as sf
from transformers import AutoProcessor, AutoModelForCTC
from pathlib import Path
import torch

def user_audio_matrix(user_audio_paths):
    
    user_audio = []
    expected_sample_rate = 16000
        
    for audio_path in user_audio_paths:
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        
        if sample_rate != expected_sample_rate:
            raise ValueError("sample rate not 16000, wav2vec2 expects 16000")
        
        user_audio.append(waveform)
        
    wav2vec2_path = Path(__file__).resolve().parents[2] / "wav2vec2"
        

    
    processor = AutoProcessor.from_pretrained(wav2vec2_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(wav2vec2_path, local_files_only=True)
    
    processed_user_audio = processor(
        user_audio,
        sample_rate=expected_sample_rate,
        return_tensor="pt",
        padding=True
    )
    
    with torch.no_grad():
        model.eval()
        output = model(**processed_user_audio)
        
    output_logits = output.logits
    output_softmax = torch.softmax(output_logits, axis=-1)
    
    return output_softmax