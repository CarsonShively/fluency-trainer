def pad_target(target_phones, target_classes):
    
    max_len = 0
    
    for sample in target_phones:
        max_len = max(max_len, len(target_phones))
    
    new_target_phones = []
    
    for sample in target_phones:
        
        new_sample = []
            
        for phone in sample:
            new_sample.append(phone)
        while len(new_sample) < max_len:
            new_sample.append(0)

        new_target_phones.append(new_sample)
        
    
    new_target_classes = []
    
    for sample in target_classes:
        
        new_sample = []
            
        for c in sample:
            new_sample.append(c)
        while len(new_sample) < max_len:
            new_sample.append(-1)

        new_target_classes.append(new_sample)
        
    return new_target_phones, new_target_classes, max_len