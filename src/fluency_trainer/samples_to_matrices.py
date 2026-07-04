import tensorflow as tf

def samples_to_matrices(train_samples, test_samples):
    train_user_audio = []
    train_target_phones = []
    train_target_classes = []
    
    test_user_audio = []
    test_target_phones = []
    test_target_classes = []
    
    for sample in train_samples.values():
        train_user_audio.append(sample["user_audio_path"])
        train_target_phones.append(sample["target_phones"])
        train_target_classes.append(sample["target_classes"])
        
    for sample in test_samples.values():
        test_user_audio.append(sample["user_audio_path"])
        test_target_phones.append(sample["target_phones"])
        test_target_classes.append(sample["target_classes"])
        
    return train_user_audio, train_target_phones, train_target_classes, test_user_audio, test_target_phones, test_target_classes