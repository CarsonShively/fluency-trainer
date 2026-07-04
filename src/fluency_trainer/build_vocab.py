def build_vocab(train_samples):
    
    vocab = {}
    vocab_id = 0
    
    vocab["<PAD>"] = vocab_id
    vocab_id += 1
    vocab["<UNK>"] = vocab_id
    vocab_id += 1
    
    for sample in train_samples.values():
        for phone in sample["target_phones"]:
            
            if phone not in vocab:
                vocab[phone] = vocab_id
                vocab_id += 1
                
    return vocab, len(vocab)

