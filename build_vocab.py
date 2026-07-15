import json
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download, HfApi, get_token

def build_vocab():
    print("building vocab...")
    train_samples_path = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns="samples/train.json"
    ))
    
    with open(train_samples_path, "r") as con:
        train_samples = json.load(con) 
    
    vocab = {}
    vocab_id = 0
    
    vocab["<PAD>"] = vocab_id
    vocab_id += 1
    vocab["<UNK>"] = vocab_id
    vocab_id += 1
    
    for sample in train_samples.values():
        for phone in sample["target_phonemes"]:
            
            if phone not in vocab:
                vocab[phone] = vocab_id
                vocab_id += 1
                
    out_path = Path(__file__).resolve().parents[0] / "vocab"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(out_path / "vocab.json", "w") as con:
        json.dump(vocab, con)
        
    print("vocab saved.")
    
    if get_token() != None:
        api = HfApi()
        api.upload_folder(
            folder_path=out_path,
            path_in_repo="vocab",
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="dataset",
            delete_patterns="vocab/**"
        )
    
if __name__ == "__main__":
    build_vocab()