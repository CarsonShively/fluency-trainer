def target_phones_matrix(target_phones, vocab):
    
    target_phones_matrix = []
    
    for sample in target_phones:
        sample_target_phone_ids = []
        for phone in sample:
            if phone in vocab:
                sample_target_phone_ids.append(vocab[phone])
            else:
                sample_target_phone_ids.append(vocab["<UNK>"])
        target_phones_matrix.append(sample_target_phone_ids)
        
    return target_phones_matrix