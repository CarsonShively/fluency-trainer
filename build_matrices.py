import json
from pathlib import Path
import numpy as np
import shutil
from huggingface_hub import snapshot_download, get_token, HfApi

def build_matrices():
   
    local_data = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns=["samples/**", "vocab/**", "user_audio/**"]
    ))
        
    print("data loaded to local")
        
    print("user audio loaded to local")
    
    print("Starting...")
    
    out_path = Path(__file__).resolve().parents[0] / "matrices"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(local_data / "vocab/vocab.json", "r") as con:
        vocab = json.load(con)
    
    with open(local_data / "samples/train.json", "r") as con:
        train_samples = json.load(con)
        
    with open(local_data / "samples/val.json", "r") as con:
        val_samples = json.load(con)
        
    with open(local_data / "samples/test.json", "r") as con:
        test_samples = json.load(con)
    
    print("loads complete")
    
    train_user_audio = []
    train_target_phones = []
    train_scores = []
    
    train_audio_path = local_data / "user_audio" / "train"
    
    for sample_id, sample in train_samples.items():
        audio = np.load(train_audio_path / f"{sample_id}.npy")
        train_user_audio.append(audio)
        
        target_phones_ids = []
        
        for phone in sample["target_phonemes"]:
            if phone in vocab:
                target_phones_ids.append(vocab[phone])
            else:
                target_phones_ids.append(vocab["<UNK>"])
        
        target_phones_numpy = np.array(target_phones_ids)
        scores_numpy = np.array(sample["scores"])
        
        train_target_phones.append(target_phones_numpy)
        train_scores.append(scores_numpy)
        
    val_user_audio = []
    val_target_phones = []
    val_scores = []
    
    val_audio_path = local_data / "user_audio" / "val"
    
    for sample_id, sample in val_samples.items():
        audio = np.load(val_audio_path / f"{sample_id}.npy")
        val_user_audio.append(audio)
        
        target_phones_ids = []
        
        for phone in sample["target_phonemes"]:
            if phone in vocab:
                target_phones_ids.append(vocab[phone])
            else:
                target_phones_ids.append(vocab["<UNK>"])
        
        target_phones_numpy = np.array(target_phones_ids)
        scores_numpy = np.array(sample["scores"])
        
        val_target_phones.append(target_phones_numpy)
        val_scores.append(scores_numpy)
            

    test_user_audio = []
    test_target_phones = []
    test_scores = []
    
    test_audio_path = local_data / "user_audio" / "test"
    
    for sample_id, sample in test_samples.items():
        audio = np.load(test_audio_path / f"{sample_id}.npy")
        test_user_audio.append(audio)
        
        target_phones_ids = []
        
        for phone in sample["target_phonemes"]:
            if phone in vocab:
                target_phones_ids.append(vocab[phone])
            else:
                target_phones_ids.append(vocab["<UNK>"])
        
        target_phones_numpy = np.array(target_phones_ids)
        scores_numpy = np.array(sample["scores"])
        
        test_target_phones.append(target_phones_numpy)
        test_scores.append(scores_numpy)
        
    train_audio_pad_len = 0
    train_phones_pad_len = 0
    
    val_audio_pad_len = 0
    val_phones_pad_len = 0
    
    test_audio_pad_len = 0
    test_phones_pad_len = 0
    
    for sample in train_user_audio:
        train_audio_pad_len = max(train_audio_pad_len, sample.shape[0])
    
    for sample in val_user_audio:
        val_audio_pad_len = max(val_audio_pad_len, sample.shape[0])
        
    for sample in test_user_audio:
        test_audio_pad_len = max(test_audio_pad_len, sample.shape[0])
        
        
        
    for sample in train_target_phones:
        train_phones_pad_len = max(train_phones_pad_len, sample.shape[0])
    
    for sample in val_target_phones:
        val_phones_pad_len = max(val_phones_pad_len, sample.shape[0])
        
    for sample in test_target_phones:
        test_phones_pad_len = max(test_phones_pad_len, sample.shape[0])
    
    train_audio_padded = []
    val_audio_padded = []
    test_audio_padded = []
    
    train_phones_padded = []
    val_phones_padded = []
    test_phones_padded = []
    
    for sample in train_user_audio:
        padded = np.zeros((train_audio_pad_len, sample.shape[1]), dtype=np.float32)
        padded[:sample.shape[0]] = sample
        train_audio_padded.append(padded)
        
    del train_user_audio
        
    for sample in val_user_audio:
        padded = np.zeros((val_audio_pad_len, sample.shape[1]), dtype=np.float32)
        padded[:sample.shape[0]] = sample
        val_audio_padded.append(padded)
        
    del val_user_audio
        
    for sample in test_user_audio:
        padded = np.zeros((test_audio_pad_len, sample.shape[1]), dtype=np.float32)
        padded[:sample.shape[0]] = sample
        test_audio_padded.append(padded)
        
    del test_user_audio
        
    for sample in train_target_phones:
        padded = np.zeros(train_phones_pad_len, dtype=np.int32)
        padded[:sample.shape[0]] = sample
        train_phones_padded.append(padded)
        
    del train_target_phones
        
    for sample in val_target_phones:
        padded = np.zeros(val_phones_pad_len, dtype=np.int32)
        padded[:sample.shape[0]] = sample
        val_phones_padded.append(padded)
       
    del val_target_phones
        
    for sample in test_target_phones:
        padded = np.zeros(test_phones_pad_len, dtype=np.int32)
        padded[:sample.shape[0]] = sample
        test_phones_padded.append(padded)
        
    del test_target_phones
     
    train_audio_numpy = np.array(train_audio_padded)
    del train_audio_padded
    val_audio_numpy = np.array(val_audio_padded)
    del val_audio_padded
    test_audio_numpy = np.array(test_audio_padded)
    del test_audio_padded
    
    train_phones_numpy = np.array(train_phones_padded)
    del train_phones_padded
    val_phones_numpy = np.array(val_phones_padded)
    del val_phones_padded
    test_phones_numpy = np.array(test_phones_padded)
    del test_phones_padded
    
    train_scores_numpy = np.array(train_scores)
    del train_scores
    val_scores_numpy = np.array(val_scores)
    del val_scores
    test_scores_numpy = np.array(test_scores)
    del test_scores
        
    train_audio_mask = np.any(train_audio_numpy != 0.0, axis=-1).astype(np.float32)
    val_audio_mask = np.any(val_audio_numpy != 0.0, axis=-1).astype(np.float32)
    test_audio_mask = np.any(test_audio_numpy != 0.0, axis=-1).astype(np.float32)
    
    train_phones_mask = (train_phones_numpy != 0).astype(np.int32)
    val_phones_mask = (val_phones_numpy != 0).astype(np.int32)
    test_phones_mask = (test_phones_numpy != 0).astype(np.int32)     
    
    np.save(out_path / "train_user_audio.npy", train_audio_numpy)
    np.save(out_path / "val_user_audio.npy", val_audio_numpy)
    np.save(out_path / "test_user_audio.npy", test_audio_numpy)
    
    np.save(out_path / "train_target_phonemes.npy", train_phones_numpy)
    np.save(out_path / "val_target_phonemes.npy", val_phones_numpy)
    np.save(out_path / "test_target_phonemes.npy", test_phones_numpy)
    
    np.save(out_path / "train_scores.npy", train_scores_numpy)
    np.save(out_path / "val_scores.npy", val_scores_numpy)
    np.save(out_path / "test_scores.npy", test_scores_numpy)

    np.save(out_path / "train_user_audio_mask.npy", train_audio_mask)
    np.save(out_path / "val_user_audio_mask.npy", val_audio_mask)
    np.save(out_path / "test_user_audio_mask.npy", test_audio_mask)

    print("matrices saved.")

    print(f"train user audio: {train_audio_numpy.shape}")
    print(f"val user audio: {val_audio_numpy.shape}")
    print(f"test user audio: {test_audio_numpy.shape}")

    print(f"train target phonemes: {train_phones_numpy.shape}")
    print(f"val target phonemes: {val_phones_numpy.shape}")
    print(f"test target phonemes: {test_phones_numpy.shape}")

    print(f"train scores: {train_scores_numpy.shape}")
    print(f"val scores: {val_scores_numpy.shape}")
    print(f"test scores: {test_scores_numpy.shape}")
    
    print(f"train user audio mask: {train_audio_mask.shape}")
    print(f"val user audio mask: {val_audio_mask.shape}")
    print(f"test user audio mask: {test_audio_mask.shape}")

    print(f"train target phonemes mask: {train_phones_mask.shape}")
    print(f"val target phonemes mask: {val_phones_mask.shape}")
    print(f"test target phonemes mask: {test_phones_mask.shape}")

    
    if get_token() is not None:
        api = HfApi()
        api.upload_folder(
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="dataset",
            folder_path=out_path,
            path_in_repo="matrices",
            delete_patterns="matrices/**"
        )
        
        
    
if __name__ == "__main__":
    build_matrices()