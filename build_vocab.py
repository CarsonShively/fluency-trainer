import json
from pathlib import Path
import shutil

def build_vocab():
    print("building vocab...")
    train_samples_path = Path(__file__).resolve().parents[0] / "data/samples/train.json"
    
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
                
    out_path = Path(__file__).resolve().parents[0] / "data/vocab"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(out_path / "vocab.json", "w") as con:
        json.dump(vocab, con)
        
    print("vocab saved.")
    
if __name__ == "__main__":
    build_vocab()