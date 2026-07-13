from pathlib import Path
import json
import shutil
from huggingface_hub import snapshot_download, HfApi, get_token
def build_samples():
    
    data_path = Path(__file__).resolve().parents[0] / "data"
    
    snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        local_dir=data_path,
        allow_patterns="speechocean762/**"
    )
    
    train_wav_path = Path(__file__).resolve().parents[0] / "data/speechocean762/train/wav.scp"
    test_wav_path = Path(__file__).resolve().parents[0] / "data/speechocean762/test/wav.scp"
    
    train_samples = {}
    val_samples = {}
    test_samples = {}
    
    train_len = len(train_wav_path.read_text().splitlines())
    
    scores_path = Path(__file__).resolve().parents[0] / "data/speechocean762/resource/scores.json"
    
    with open(scores_path, "r") as file:
        scores_dict = json.load(file)
    
    with open(train_wav_path, "r") as file:
        
        val_split_index = int(train_len * 0.90)
        sample_number = 0
        for line in file:
            parts = line.strip().split()
            
            sample_id = parts[0]
            user_audio_path = f"data/speechocean762/{parts[1]}"
            
            
            
            target_phonemes = []
            scores = []
            
            scores.append(scores_dict[sample_id]["accuracy"])
            scores.append(scores_dict[sample_id]["fluency"])
            scores.append(scores_dict[sample_id]["prosodic"])
            scores.append(scores_dict[sample_id]["total"])
            
            for word in scores_dict[sample_id]["words"]:
                target_phonemes.extend(word["phones"])
                    
            sample = {
                "user_audio_path": str(user_audio_path),
                "target_phonemes": target_phonemes,
                "scores": scores
            }
            
            if sample_number < val_split_index:
                train_samples[sample_id] = sample
                print(f"train sample: {sample_number}")
            else:
                val_samples[sample_id] = sample
                print(f"val sample: {sample_number}")                

            sample_number += 1
    
    print("train and val samples complete.")
        
    with open(test_wav_path, "r") as file:
        sample_number = 0
        for line in file:
            parts = line.strip().split()
            
            sample_id = parts[0]
            user_audio_path = f"data/speechocean762/{parts[1]}"
            
            target_phonemes = []
            scores = []
            
            scores.append(scores_dict[sample_id]["accuracy"])
            scores.append(scores_dict[sample_id]["fluency"])
            scores.append(scores_dict[sample_id]["prosodic"])
            scores.append(scores_dict[sample_id]["total"])
                
            for word in scores_dict[sample_id]["words"]:
                target_phonemes.extend(word["phones"])
                    
            sample = {
                "user_audio_path": str(user_audio_path),
                "target_phonemes": target_phonemes,
                "scores": scores
            }
            
            test_samples[sample_id] = sample
            print(f"test sample: {sample_number}")
            sample_number += 1
            
    print("test samples complete")
    
    out_path = Path(__file__).resolve().parents[0] / "data/samples"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(out_path / "train.json", "w") as con:
        json.dump(train_samples, con)
        
    with open(out_path / "val.json", "w") as con:
        json.dump(val_samples, con)
        
    with open(out_path / "test.json", "w") as con:
        json.dump(test_samples, con)
    
    
    if get_token() is not None:
        api = HfApi()
        api.upload_folder(
            folder_path=out_path,
            path_in_repo="samples",
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="dataset",
            delete_patterns="*"
        )
        print("samples uploaded.")
    else:
        print("samples local")
    
if __name__ == "__main__":
    build_samples()