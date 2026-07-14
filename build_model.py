from fluency_trainer.phone_audio_alignment_model import PhoneAudioAlignmentModel
from fluency_trainer.train import train

import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
import json
from huggingface_hub import snapshot_download, HfApi, get_token


def build_model():
    
    local_data = Path(__file__).resolve().parents[0] / "data"
    
    snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns=["matrices/train/**", "matrices/val/**", "vocab/**"],
        local_dir=local_data
    )
    
    with open(local_data / "vocab/vocab.json", "r") as con:
        vocab = json.load(con)
    
    vocab_len = len(vocab)
    print(f"vocab len: {vocab_len}")
    
    del vocab
    
    train_user_audio = np.load(local_data / "matrices/train_user_audio.npy")
    val_user_audio = np.load(local_data / "matrices/val_user_audio.npy")

    train_user_audio_mask = np.load(local_data / "matrices/train_user_audio_mask.npy")
    val_user_audio_mask = np.load(local_data / "matrices/val_user_audio_mask.npy")

    train_target_phonemes = np.load(local_data / "matrices/train_target_phonemes.npy")
    val_target_phonemes = np.load(local_data / "matrices/val_target_phonemes.npy")

    train_target_phonemes_mask = np.load(local_data / "matrices/train_target_phonemes_mask.npy")
    val_target_phonemes_mask = np.load(local_data / "matrices/val_target_phonemes_mask.npy")
    
    train_scores = np.load(local_data / "matrices/train_target_classes.npy")
    val_scores = np.load(local_data / "matrices/val_target_classes.npy")

    print("point 1")
    
    train_user_audio_tensor = tf.convert_to_tensor(train_user_audio, dtype=tf.float32)
    del train_user_audio
    val_user_audio_tensor = tf.convert_to_tensor(val_user_audio, dtype=tf.float32)
    del val_user_audio

    train_user_audio_mask_tensor = tf.convert_to_tensor(train_user_audio_mask, dtype=tf.float32)
    del train_user_audio_mask
    val_user_audio_mask_tensor = tf.convert_to_tensor(val_user_audio_mask, dtype=tf.float32)
    del val_user_audio_mask

    train_target_phonemes_tensor = tf.convert_to_tensor(train_target_phonemes, dtype=tf.int32)
    del train_target_phonemes
    val_target_phonemes_tensor = tf.convert_to_tensor(val_target_phonemes, dtype=tf.int32)
    del val_target_phonemes

    train_target_phonemes_mask_tensor = tf.convert_to_tensor(train_target_phonemes_mask, dtype=tf.int32)
    del train_target_phonemes_mask
    val_target_phonemes_mask_tensor = tf.convert_to_tensor(val_target_phonemes_mask, dtype=tf.int32)
    del val_target_phonemes_mask
    
    train_target_classes_tensor = tf.convert_to_tensor(train_scores, dtype=tf.float32)
    del train_scores
    val_target_classes_tensor = tf.convert_to_tensor(val_scores, dtype=tf.float32)
    del val_scores

    
    print("point 2")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": train_user_audio_tensor,
            "user_audio_mask": train_user_audio_mask_tensor,
            "target_phones": train_target_phonemes_tensor,
            "target_phones_mask": train_target_phonemes_mask_tensor
        },
        train_target_classes_tensor
    ))
    
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": val_user_audio_tensor,
            "user_audio_mask": val_user_audio_mask_tensor,
            "target_phones": val_target_phonemes_tensor,
            "target_phones_mask": val_target_phonemes_mask_tensor
        },
        val_target_classes_tensor,
    ))
    print("point 3")
    train_dataset_batched = train_dataset.shuffle(buffer_size=1000).batch(16)
    val_dataset_batched = val_dataset.batch(16)
    
    print("point 5")
    
    
    
    model = PhoneAudioAlignmentModel(vocab_size=vocab_len)
    print("point 6")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    
    loss_fn = tf.losses.MeanSquaredError()
    print("point 7")
    val_loss = train(train_dataset=train_dataset_batched, val_dataset=val_dataset_batched, model=model, optimizer=optimizer, loss_fn=loss_fn)
    
    print(f"val loss: {val_loss}")
    
    
    out_path = Path(__file__).resolve().parents[0] / "fluency_trainer_model"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    model.save_weights(out_path / "phone_audio_alignment_model.weights.h5")
    
    api = HfApi()

    if get_token() != None:
        api.upload_folder(
            folder_path=out_path,
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="model",
            path_in_repo="phone_audio_alignment_model",
            delete_patterns="*"   
        )
    
if __name__ == "__main__":
    build_model()
    
