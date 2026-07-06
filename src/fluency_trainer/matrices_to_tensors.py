import tensorflow as tf

def matrices_to_tensors(user_audio, target_phones, target_classes):
    user_audio_tensor = tf.convert_to_tensor(user_audio.cpu().numpy(), dtype=tf.float32)
    target_phones_tensor = tf.convert_to_tensor(target_phones, dtype=tf.float32)
    target_classes_tensor = tf.convert_to_tensor(target_classes, dtype=tf.float32)
    
    target_phones_mask_tensor = tf.not_equal(target_phones_tensor, 0)
    target_classes_mask_tensor = tf.not_equal(target_classes_tensor, -1)
    
    return user_audio_tensor, target_phones_tensor, target_classes_tensor, target_phones_mask_tensor, target_classes_mask_tensor