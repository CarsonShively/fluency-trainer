from transformers import AutoProcessor, AutoModelForCTC
import torch
import json
from pathlib import Path
import soundfile as sf
import numpy as np
import shutil

def build_matrices():
    print("Starting...")
    
    out_path = Path(__file__).resolve().parents[0] / "data/matrices"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
    
    expected_sample_rate = 16000
    
    wav2vec2_path = Path(__file__).resolve().parents[0] / "wav2vec2"
    
    processor = AutoProcessor.from_pretrained(wav2vec2_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(wav2vec2_path, local_files_only=True)
    
    vocab_path = Path(__file__).resolve().parents[0] / "data/vocab"
    with open(vocab_path / "vocab.json", "r") as con:
        vocab = json.load(con)
    
    samples_path = Path(__file__).resolve().parents[0] / "data/samples"
    
    with open(samples_path / "train.json", "r") as con:
        train_samples = json.load(con)
        
    with open(samples_path / "val.json", "r") as con:
        val_samples = json.load(con)
        
    with open(samples_path / "test.json", "r") as con:
        test_samples = json.load(con)
    
    print("loads complete")
    
    train_user_audio = []
    train_target_phones = []
    train_target_classes = []
    
    val_user_audio = []
    val_target_phones = []
    val_target_classes = []
    
    test_user_audio = []
    test_target_phones = []
    test_target_classes = []
    
    samples = (
        (train_samples, train_user_audio, train_target_phones, train_target_classes), 
        (val_samples, val_user_audio, val_target_phones, val_target_classes), 
        (test_samples, test_user_audio, test_target_phones, test_target_classes)
        )
    loop_counter = 1
    for split, audio, phones, classes in samples:
        sample_counter = 0
        for sample in split.values():
            audio_path = Path(__file__).resolve().parents[0] / sample["user_audio_path"]
            waveform, sample_rate = sf.read(audio_path)
            
            if sample_rate != expected_sample_rate:
                raise ValueError(f"sample rate: {sample_rate}, expected sample rate: {expected_sample_rate}")
            
            processed_user_audio = processor(
                waveform,
                sampling_rate=expected_sample_rate,
                return_tensors="pt"
            )
            
            model.eval()
            with torch.no_grad():
                output = model(**processed_user_audio)
                
            output_logits = output.logits
            user_audio_append = output_logits.squeeze(0).cpu().numpy()
            
            target_phones_append = []
            
            for phone in sample["target_phonemes"]:
                if phone in vocab:
                    target_phones_append.append(vocab[phone])
                else:
                    target_phones_append.append(vocab["<UNK>"])
                    
            audio.append(user_audio_append)
            phones.append(target_phones_append)
            classes.append(sample["target_classes"])
            
            sample_counter += 1
            if loop_counter == 1:
                print(f"train sample complete: {sample_counter}")
            elif loop_counter == 2:
                print(f"val sample complete: {sample_counter}")
            else:
                print(f"test sample complete: {sample_counter}") 
        
        loop_counter += 1
    print("initial matrices complete.")
            
    audio_matrices = (train_user_audio, val_user_audio, test_user_audio)
    
    audio_max_len = 0
    
    for audio_matrix in audio_matrices:
        for audio in audio_matrix:
            audio_max_len = max(audio_max_len, audio.shape[0])
            
    target_phones_matrices = (train_target_phones, val_target_phones, test_target_phones)
    
    target_phones_max_len = 0
    
    for matrix in target_phones_matrices:
        for phones in matrix:
            target_phones_max_len = max(target_phones_max_len, len(phones))
            
    print("lengths esstablished")
    
    train_padded_audio = []
    val_padded_audio = []
    test_padded_audio = []
    
    padded_audio_matrices = ((train_user_audio, train_padded_audio), (val_user_audio, val_padded_audio), (test_user_audio, test_padded_audio))
    
    for audio_matrix, append_padded_audio in padded_audio_matrices:
        for audio in audio_matrix:
            padded_array = np.zeros((audio_max_len, audio.shape[1]), dtype=audio.dtype)
            padded_array[:audio.shape[0]] = audio
            append_padded_audio.append(padded_array)
            
    train_padded_phones = []
    val_padded_phones = []
    test_padded_phones = []
    
    padded_phone_matrices = ((train_target_phones, train_padded_phones), (val_target_phones, val_padded_phones), (test_target_phones, test_padded_phones))
    
    for phones_matrix, append_padded_phones in padded_phone_matrices:
        for phones in phones_matrix:
            to_append = []
            for phone in phones:
                to_append.append(phone)
            while len(to_append) < target_phones_max_len:
                to_append.append(vocab["<PAD>"])
                
            append_padded_phones.append(to_append)
            
    train_padded_classes = []
    val_padded_classes = []
    test_padded_classes = []
    
    padded_classes_matrices = ((train_target_classes, train_padded_classes), (val_target_classes, val_padded_classes), (test_target_classes, test_padded_classes))
    
    for classes_matrix, append_padded_classes in padded_classes_matrices:
        for classes in classes_matrix:
            to_append = []
            for c in classes:
                to_append.append(c)
            while len(to_append) < target_phones_max_len:
                to_append.append(-1)
                
            append_padded_classes.append(to_append)
            
    print("matrices padded.")
            
    train_audio_numpy = np.array(train_padded_audio).astype(np.float32)
    val_audio_numpy = np.array(val_padded_audio).astype(np.float32)
    test_audio_numpy = np.array(test_padded_audio).astype(np.float32)
    
    train_phones_numpy = np.array(train_padded_phones).astype(np.int32)
    val_phones_numpy = np.array(val_padded_phones).astype(np.int32)
    test_phones_numpy = np.array(test_padded_phones).astype(np.int32)
    
    train_classes_numpy = np.array(train_padded_classes).astype(np.int32)
    val_classes_numpy = np.array(val_padded_classes).astype(np.int32)
    test_classes_numpy = np.array(test_padded_classes).astype(np.int32)
    
    train_audio_mask = np.any(train_audio_numpy != 0.0, axis=-1).astype(np.float32)
    val_audio_mask = np.any(val_audio_numpy != 0.0, axis=-1).astype(np.float32)
    test_audio_mask = np.any(test_audio_numpy != 0.0, axis=-1).astype(np.float32)
    
    train_phones_mask = (train_phones_numpy != 0).astype(np.float32)
    val_phones_mask = (val_phones_numpy != 0).astype(np.float32)
    test_phones_mask = (test_phones_numpy != 0).astype(np.float32)
    
    train_classes_mask = (train_classes_numpy != -1).astype(np.float32)
    val_classes_mask = (val_classes_numpy != -1).astype(np.float32)
    test_classes_mask = (test_classes_numpy != -1).astype(np.float32)
    
    print(f"train user audio: {train_audio_numpy.shape}")
    print(f"val user audio: {val_audio_numpy.shape}")
    print(f"test user audio: {test_audio_numpy.shape}")
    
    print(f"train target phonemes: {train_phones_numpy.shape}")
    print(f"val target phonemes: {val_phones_numpy.shape}")
    print(f"test target phonemes: {test_phones_numpy.shape}")
    
    print(f"train target classes: {train_classes_numpy.shape}")
    print(f"val target classes: {val_classes_numpy.shape}")
    print(f"test target classes: {test_classes_numpy.shape}")
    
    print(f"train user audio mask: {train_audio_mask.shape}")
    print(f"val user audio mask: {val_audio_mask.shape}")
    print(f"test user audio mask: {test_audio_mask.shape}")
    
    print(f"train target phonemes mask: {train_phones_mask.shape}")
    print(f"val target phonemes mask: {val_phones_mask.shape}")
    print(f"test target phonemes mask: {test_phones_mask.shape}")
    
    print(f"train target classes mask: {train_classes_mask.shape}")
    print(f"val target classes mask: {val_classes_mask.shape}")
    print(f"test target classes mask: {test_classes_mask.shape}")
    
    out_path.mkdir(parents=True, exists_ok=True)
    
    np.save(out_path / "train_user_audio.npy", train_audio_numpy)
    np.save(out_path / "val_user_audio.npy", val_audio_numpy) 
    np.save(out_path / "test_user_audio.npy", test_audio_numpy)
        
    np.save(out_path / "train_target_phones.npy", train_phones_numpy)
    np.save(out_path / "val_target_phones.npy", val_phones_numpy)
    np.save(out_path / "test_target_phones.npy", test_phones_numpy)
        
    np.save(out_path / "train_target_classes.npy", train_classes_numpy)
    np.save(out_path / "val_target_classes.npy", val_classes_numpy)
    np.save(out_path / "test_target_classes.npy", test_classes_numpy)
        
        
    np.save(out_path / "train_user_audio_mask.npy", train_audio_mask)
    np.save(out_path / "val_user_audio_mask.npy", val_audio_mask)
    np.save(out_path / "test_user_audio_mask.npy", test_audio_mask)
        
    np.save(out_path / "train_target_phones_mask.npy", train_phones_mask)
    np.save(out_path / "val_target_phones_mask.npy", val_phones_mask)
    np.save(out_path / "test_target_phones_mask.npy", test_phones_mask)
        
    np.save(out_path / "train_target_classes_mask.npy", train_classes_mask)
    np.save(out_path / "val_target_classes_mask.npy", val_classes_mask)
    np.save(out_path / "test_target_classes_mask.npy", test_classes_mask)
    
    print("matrices saved.")
    
if __name__ == "__main__":
    build_matrices()