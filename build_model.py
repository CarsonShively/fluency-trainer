from fluency_trainer.dataloader import dataloader
from fluency_trainer.build_vocab import build_vocab
from fluency_trainer.samples_to_matrices import samples_to_matrices
from fluency_trainer.target_phones_matrix import target_phones_matrix
from fluency_trainer.user_audio_matrix import user_audio_matrix
from fluency_trainer.pad_target import pad_target
from fluency_trainer.matrices_to_tensors import matrices_to_tensors
from fluency_trainer.user_phoneme_encoder import UserPhonemeEncoder
from fluency_trainer.target_phoneme_encoder import TargetPhonemeEncoder
from fluency_trainer.cross_attention_transformer import CrossAttentionTransformer
from fluency_trainer.classifier_head import ClassifierHead

import tensorflow as tf

# from tensor slices == class method on Dataset class (returns dataset object)
# shuffle arg == default to lenght of batch
# dataset == combine by sample (left dimension)

def build_model():
    train_samples, test_samples = dataloader()
    
    vocab, vocab_size = build_vocab(train_samples=train_samples)
    
    train_user_audio, train_target_phones, train_target_classes, test_user_audio, test_target_phones, test_target_classes = samples_to_matrices(train_samples=train_samples, test_samples=test_samples)
    
    train_target_phones_matrix = target_phones_matrix(target_phones=train_target_phones, vocab=vocab)
    
    padded_train_user_audio_matrix, train_user_audio_matrix_mask, audio_max_len, uncertianty_vector_size = user_audio_matrix(user_audio_paths=train_user_audio)
    
    padded_train_target_phones_matrix, padded_train_target_classes_matrix, max_target_len = pad_target(target_phones=train_target_phones_matrix, target_classes=train_target_classes)
    
    train_user_audio_tensor, train_target_phones_tensor, train_target_classes_tensor, train_target_phones_mask_tensor, train_target_classes_mask_tensor = matrices_to_tensors(user_audio=padded_train_user_audio_matrix, target_phones=padded_train_target_phones_matrix, target_classes=padded_train_target_classes_matrix)
    train_user_audio_mask_tensor = tf.convert_to_tensor(train_user_audio_matrix_mask.cpu().numpy(), dtype=tf.float32)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": train_user_audio_tensor,
            "user_audio_mask": train_user_audio_mask_tensor,
            "target_phones": train_target_phones_tensor,
            "target_phones_mask": train_target_phones_mask_tensor
        },
        train_target_classes_tensor,
        train_target_classes_mask_tensor
    ))
    
    train_dataset_batched = train_dataset.shuffle(len(train_target_classes_tensor)).batch(16)
    
    user_audio_encoder = UserPhonemeEncoder(max_phonemes=audio_max_len, uncertianty_vector_size=uncertianty_vector_size, dense=64, dense1=128)
    target_phoneme_encoder = TargetPhonemeEncoder(vocab_size=vocab_size, max_phonemes=max_target_len, embeded_vector_size=128, dense=64, dense1=128)
    cross_attention_transformer = CrossAttentionTransformer(embeded_vector_size=128, desne=64, dense1=128)
    classifier_head = ClassifierHead(embeded_vector_size=128, desne=64, dense1=128)
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    
    # padding == only add for tensors and cant be longer than map
    # encoders can have different y lens
    # padded target == 3rd item in touple
    # sparse categorical == 1 list (not one hot cols)
    # tf adam only takes in knobs (no model)