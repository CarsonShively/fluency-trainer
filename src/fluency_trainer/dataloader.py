from pathlib import Path
import json

def dataloader():
    
    train_wav_path = Path(__file__).resolve().parents[2] / "data/speechocean762/train/wav.scp"
    test_wav_path = Path(__file__).resolve().parents[2] / "data/speechocean762/test/wav.scp"
    
    train_samples = {}
    val_samples = {}
    test_samples = {}
    
    train_len = len(train_wav_path.read_text().splitlines())
    
    with open(train_wav_path, "r") as file:
        
        val_split_index = int(train_len * 0.90)
        sample_number = 0
        for line in file:
            parts = line.strip().split()
            
            sample_id = parts[0]
            user_audio_path = Path(__file__).resolve().parents[2] / "data/speechocean762" / parts[1]
            
            scores_path = Path(__file__).resolve().parents[2] / "data/speechocean/resource/scores.json"
            
            target_phonemes = []
            target_classes = []
            
            with open(scores_path, "r") as file:
                scores_dict = json.load(file)
                
                for word in scores_dict[sample_id]["words"]:
                    target_phonemes.extend(word["phones"])
                    target_classes.extend(word["phones-accuracy"])
                    
            sample = {
                "user_audio_path": user_audio_path,
                "target_phonemes": target_phonemes,
                "target_classes": target_classes
            }
            
            if sample_number < val_split_index:
                train_samples[sample_id] = sample
            else:
                val_samples[sample_id] = sample
                
            sample_number += 1
            
    with open(test_wav_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            
            sample_id = parts[0]
            user_audio_path = Path(__file__).resolve().parents[2] / "data/speechocean762" / parts[1]
            
            scores_path = Path(__file__).resolve().parents[2] / "data/speechocean/resource/scores.json"
            
            target_phonemes = []
            target_classes = []
            
            with open(scores_path, "r") as file:
                scores_dict = json.load(file)
                
                for word in scores_dict[sample_id]["words"]:
                    target_phonemes.extend(word["phones"])
                    target_classes.extend(word["phone-accuracy"])
                    
            sample = {
                "user_audio_path": user_audio_path,
                "target_phonemes": target_phonemes,
                "target_classes": target_classes
            }
            
            test_samples[sample_id] = sample
            
    return train_samples, val_samples, test_samples