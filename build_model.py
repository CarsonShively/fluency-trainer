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
from fluency_trainer.train import train

import tensorflow as tf
from pathlib import Path
import shutil
import json

def build_model():
    train_samples, val_samples, test_samples = dataloader()
    
    vocab, vocab_size = build_vocab(train_samples=train_samples)
    
    train_user_audio, train_target_phones, train_target_classes, val_user_audio, val_target_phones, val_target_classes, test_user_audio, test_target_phones, test_target_classes = samples_to_matrices(train_samples=train_samples, val_samples=val_samples, test_samples=test_samples)
    
    train_target_phones_matrix = target_phones_matrix(target_phones=train_target_phones, vocab=vocab)
    val_target_phones_matrix = target_phones_matrix(target_phones=val_target_phones, vocab=vocab)
    test_target_phones_matrix = target_phones_matrix(target_phones=test_target_phones, vocab=vocab)
    
    padded_train_user_audio_matrix, train_user_audio_matrix_mask, audio_max_len, uncertianty_vector_size = user_audio_matrix(user_audio_paths=train_user_audio)
    padded_val_user_audio_matrix, val_user_audio_matrix_mask, audio_max_len, uncertianty_vector_size = user_audio_matrix(user_audio_paths=val_user_audio)
    padded_test_user_audio_matrix, test_user_audio_matrix_mask, audio_max_len, uncertianty_vector_size = user_audio_matrix(user_audio_paths=test_user_audio)
    
    padded_train_target_phones_matrix, padded_train_target_classes_matrix, max_target_len = pad_target(target_phones=train_target_phones_matrix, target_classes=train_target_classes)
    padded_val_target_phones_matrix, padded_val_target_classes_matrix, max_target_len = pad_target(target_phones=val_target_phones_matrix, target_classes=val_target_classes)
    padded_test_target_phones_matrix, padded_test_target_classes_matrix, max_target_len = pad_target(target_phones=test_target_phones_matrix, target_classes=test_target_classes)
    
    train_user_audio_tensor, train_target_phones_tensor, train_target_classes_tensor, train_target_phones_mask_tensor, train_target_classes_mask_tensor = matrices_to_tensors(user_audio=padded_train_user_audio_matrix, target_phones=padded_train_target_phones_matrix, target_classes=padded_train_target_classes_matrix)
    val_user_audio_tensor, val_target_phones_tensor, val_target_classes_tensor, val_target_phones_mask_tensor, val_target_classes_mask_tensor = matrices_to_tensors(user_audio=padded_val_user_audio_matrix, target_phones=padded_val_target_phones_matrix, target_classes=padded_val_target_classes_matrix)
    test_user_audio_tensor, test_target_phones_tensor, test_target_classes_tensor, test_target_phones_mask_tensor, test_target_classes_mask_tensor = matrices_to_tensors(user_audio=padded_test_user_audio_matrix, target_phones=padded_test_target_phones_matrix, target_classes=padded_test_target_classes_matrix)
    train_user_audio_mask_tensor = tf.convert_to_tensor(train_user_audio_matrix_mask.cpu().numpy(), dtype=tf.float32)
    val_user_audio_mask_tensor = tf.convert_to_tensor(val_user_audio_matrix_mask.cpu().numpy(), dtype=tf.float32)
    test_user_audio_mask_tensor = tf.convert_to_tensor(test_user_audio_matrix_mask.cpu().numpy(), dtype=tf.float32)
    
    
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
    
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": val_user_audio_tensor,
            "user_audio_mask": val_user_audio_mask_tensor,
            "target_phones": val_target_phones_tensor,
            "target_phones_mask": val_target_phones_mask_tensor
        },
        val_target_classes_tensor,
        val_target_classes_mask_tensor
    ))
        
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": test_user_audio_tensor,
            "user_audio_mask": test_user_audio_mask_tensor,
            "target_phones": test_target_phones_tensor,
            "target_phones_mask": test_target_phones_mask_tensor
        },
        test_target_classes_tensor,
        test_target_classes_mask_tensor
    ))
    
    train_dataset_batched = train_dataset.shuffle(len(train_target_classes_tensor)).batch(16)
    val_dataset_batched = val_dataset.shuffle(len(val_target_classes_tensor)).batch(16)
    test_dataset_batched = test_dataset.shuffle(len(test_target_classes_tensor)).batch(16)
    
    user_audio_encoder = UserPhonemeEncoder(max_phonemes=audio_max_len, uncertianty_vector_size=uncertianty_vector_size, dense=64, dense1=128)
    target_phoneme_encoder = TargetPhonemeEncoder(vocab_size=vocab_size, max_phonemes=max_target_len, embeded_vector_size=128, dense=64, dense1=128)
    cross_attention_transformer = CrossAttentionTransformer(embeded_vector_size=128, desne=64, dense1=128)
    classifier_head = ClassifierHead(embeded_vector_size=128, desne=64, dense1=128)
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    
    val_loss = train(train_dataset=train_dataset_batched, val_dataset=val_dataset_batched, user_encoder=user_audio_encoder, target_encoder=target_phoneme_encoder, transformer=cross_attention_transformer, classifier=classifier_head, optimizer=optimizer, loss_fn=loss_fn)
    
    total_holdout_loss = 0
    batch_counter = 0
    for batch, classes, mask in test_dataset_batched:
        user_attention = user_audio_encoder(batch["user_audio"], batch["user_audio_mask"])
        target_attention = target_phoneme_encoder(batch["target_phones"], batch["target_phones_mask"])
        cross_attention = cross_attention_transformer(user_attention, target_attention, batch["user_audio_mask"], batch["target_phones_mask"])
        logits = classifier_head(cross_attention)
        loss = loss_fn(logits, classes)
        masked_loss = loss * mask
        avg_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
        total_holdout_loss += avg_loss
        batch_counter += 1
        
    holdout_loss = total_holdout_loss / batch_counter
    
    print(f"holdout loss: {holdout_loss}")
    
    out_path = Path(__file__).resolve().parents[0] / "model_artifacts"
    if out_path.is_dir():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    user_audio_encoder.save_weights(out_path / "audio_encoder.weights.h5")
    target_phoneme_encoder.save_weights(out_path / "phone_encoder.weights.h5")
    cross_attention_transformer.save_weights(out_path / "cross_attention_transformer.weights.h5")
    classifier_head.save_weights(out_path / "classifier_head_weights.h5")
    
    metrics = {
        "val_loss": val_loss,
        "holdout_loss": holdout_loss
    }
    
    with open(out_path / "metrics.json", "w") as con:
        json.dump(metrics, con)
        
