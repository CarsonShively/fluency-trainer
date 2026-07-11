from fluency_trainer.user_phoneme_encoder import UserPhonemeEncoder
from fluency_trainer.target_phoneme_encoder import TargetPhonemeEncoder
from fluency_trainer.cross_attention_transformer import CrossAttentionTransformer
from fluency_trainer.classifier_head import ClassifierHead
from fluency_trainer.train import train

import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
import json
from huggingface_hub import snapshot_download

# snapshot == function that executes and returns nothing
# masks float vs int
# data.from tensor slices == takes in touple 
# load train and val free then load test
# go lower in the backprop?
# only shuffle train batches
# cache here tracks not stores (doesnt duplicate)
# google drive cache?
# hf local vs cache?
# snapshot no local dir == returns a string cache path (put inside Path() to return path object) 

def build_model():
    
    cache_data = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns="matrices/**"
    ))
    
    train_user_audio = np.load(cache_data / "matrices/train_user_audio.npy")
    val_user_audio = np.load(cache_data / "matrices/val_user_audio.npy")
    test_user_audio = np.load(cache_data / "matrices/test_user_audio.npy")

    train_user_audio_mask = np.load(cache_data / "matrices/train_user_audio_mask.npy")
    val_user_audio_mask = np.load(cache_data / "matrices/val_user_audio_mask.npy")
    test_user_audio_mask = np.load(cache_data / "matrices/test_user_audio_mask.npy")

    train_target_phonemes = np.load(cache_data / "matrices/train_target_phonemes.npy")
    val_target_phonemes = np.load(cache_data / "matrices/val_target_phonemes.npy")
    test_target_phonemes = np.load(cache_data / "matrices/test_target_phonemes.npy")

    train_target_phonemes_mask = np.load(cache_data / "matrices/train_target_phonemes_mask.npy")
    val_target_phonemes_mask = np.load(cache_data / "matrices/val_target_phonemes_mask.npy")
    test_target_phonemes_mask = np.load(cache_data / "matrices/test_target_phonemes_mask.npy")
    
    
    train_target_classes = np.load(cache_data / "matrices/train_target_classes.npy")
    val_target_classes = np.load(cache_data / "matrices/val_target_classes.npy")
    test_target_classes = np.load(cache_data / "matrices/test_target_classes.npy")

    train_target_classes_mask = np.load(cache_data / "matrices/train_target_classes_mask.npy")
    val_target_classes_mask = np.load(cache_data / "matrices/val_target_classes_mask.npy")
    test_target_classes_mask = np.load(cache_data / "matrices/test_target_classes_mask.npy")
    
    print("point 1")
    
    train_user_audio_tensor = tf.convert_to_tensor(train_user_audio, dtype=tf.float32)
    val_user_audio_tensor = tf.convert_to_tensor(val_user_audio, dtype=tf.float32)
    test_user_audio_tensor = tf.convert_to_tensor(test_user_audio, dtype=tf.float32)

    train_user_audio_mask_tensor = tf.convert_to_tensor(train_user_audio_mask, dtype=tf.float32)
    val_user_audio_mask_tensor = tf.convert_to_tensor(val_user_audio_mask, dtype=tf.float32)
    test_user_audio_mask_tensor = tf.convert_to_tensor(test_user_audio_mask, dtype=tf.float32)

    train_target_phonemes_tensor = tf.convert_to_tensor(train_target_phonemes, dtype=tf.int32)
    val_target_phonemes_tensor = tf.convert_to_tensor(val_target_phonemes, dtype=tf.int32)
    test_target_phonemes_tensor = tf.convert_to_tensor(test_target_phonemes, dtype=tf.int32)

    train_target_phonemes_mask_tensor = tf.convert_to_tensor(train_target_phonemes_mask, dtype=tf.int32)
    val_target_phonemes_mask_tensor = tf.convert_to_tensor(val_target_phonemes_mask, dtype=tf.int32)
    test_target_phonemes_mask_tensor = tf.convert_to_tensor(test_target_phonemes_mask, dtype=tf.int32)
    
    
    train_target_classes_tensor = tf.convert_to_tensor(train_target_classes, dtype=tf.int32)
    val_target_classes_tensor = tf.convert_to_tensor(val_target_classes, dtype=tf.int32)
    test_target_classes_tensor = tf.convert_to_tensor(test_target_classes, dtype=tf.int32)

    train_target_classes_mask_tensor = tf.convert_to_tensor(train_target_classes_mask, dtype=tf.int32)
    val_target_classes_mask_tensor = tf.convert_to_tensor(val_target_classes_mask, dtype=tf.int32)
    test_target_classes_mask_tensor = tf.convert_to_tensor(test_target_classes_mask, dtype=tf.int32)
    
    print("point 2")
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": train_user_audio_tensor,
            "user_audio_mask": train_user_audio_mask_tensor,
            "target_phones": train_target_phonemes_tensor,
            "target_phones_mask": train_target_phonemes_mask_tensor
        },
        train_target_classes_tensor,
        train_target_classes_mask_tensor
    ))
    
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": val_user_audio_tensor,
            "user_audio_mask": val_user_audio_mask_tensor,
            "target_phones": val_target_phonemes_tensor,
            "target_phones_mask": val_target_phonemes_mask_tensor
        },
        val_target_classes_tensor,
        val_target_classes_mask_tensor
    ))
        
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": test_user_audio_tensor,
            "user_audio_mask": test_user_audio_mask_tensor,
            "target_phones": test_target_phonemes_tensor,
            "target_phones_mask": test_target_phonemes_mask_tensor
        },
        test_target_classes_tensor,
        test_target_classes_mask_tensor
    ))
    
    print("point 3")
    
    train_dataset_batched = train_dataset.shuffle(len(train_target_classes_tensor)).batch(16)
    val_dataset_batched = val_dataset.batch(16)
    test_dataset_batched = test_dataset.batch(16)
    
    print("load successful")
    
    """ 
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
    """
        
if __name__ == "__main__":
    build_model()