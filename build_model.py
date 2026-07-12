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
from huggingface_hub import snapshot_download, HfApi, get_token

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
# stay in colab with hf storage (code in vscode and clone to colab to run)
# go lower in code for gradient calc and back prop? (later?)
# tensorflow == all functinos methods etc detect gpu automatially (no code change needed at all, even at lowest level of tensor flow use)
# verify target phones and classes have same len
# training dropout later
# fusion model?
def build_model():
    
    cache_data = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns=["matrices/**", "vocab/**"]
    ))
    
    with open(cache_data / "vocab/vocab.json", "r") as con:
        vocab = json.load(con)
    
    vocab_len = len(vocab)
    
    del vocab
    
    train_user_audio = np.load(cache_data / "matrices/train_user_audio.npy")
    val_user_audio = np.load(cache_data / "matrices/val_user_audio.npy")

    train_user_audio_mask = np.load(cache_data / "matrices/train_user_audio_mask.npy")
    val_user_audio_mask = np.load(cache_data / "matrices/val_user_audio_mask.npy")

    train_target_phonemes = np.load(cache_data / "matrices/train_target_phonemes.npy")
    val_target_phonemes = np.load(cache_data / "matrices/val_target_phonemes.npy")

    train_target_phonemes_mask = np.load(cache_data / "matrices/train_target_phonemes_mask.npy")
    val_target_phonemes_mask = np.load(cache_data / "matrices/val_target_phonemes_mask.npy")
    
    
    train_target_classes = np.load(cache_data / "matrices/train_target_classes.npy")
    val_target_classes = np.load(cache_data / "matrices/val_target_classes.npy")

    train_target_classes_mask = np.load(cache_data / "matrices/train_target_classes_mask.npy")
    val_target_classes_mask = np.load(cache_data / "matrices/val_target_classes_mask.npy")
    
    test_user_audio = np.load(cache_data / "matrices/test_user_audio.npy")
    test_target_phonemes = np.load(cache_data / "matrices/test_target_phonemes.npy")
    
    audio_len = max(train_user_audio.shape[1], val_user_audio.shape[1], test_user_audio.shape[1])
    target_len = max(train_target_phonemes.shape[1], val_target_phonemes.shape[1], test_target_phonemes.shape[1])
    uncertainty_vecctor_size = train_user_audio.shape[2]

    print("point 1")
    
    del test_user_audio, test_target_phonemes
    
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
    
    train_target_classes_tensor = tf.convert_to_tensor(train_target_classes, dtype=tf.int32)
    del train_target_classes
    val_target_classes_tensor = tf.convert_to_tensor(val_target_classes, dtype=tf.int32)
    del val_target_classes

    train_target_classes_mask_tensor = tf.convert_to_tensor(train_target_classes_mask, dtype=tf.int32)
    del train_target_classes_mask
    val_target_classes_mask_tensor = tf.convert_to_tensor(val_target_classes_mask, dtype=tf.int32)
    del val_target_classes_mask
    
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
    print("point 3")
    train_dataset_batched = train_dataset.shuffle(buffer_size=1000).batch(16)
    val_dataset_batched = val_dataset.batch(16)
    
    print("point 5")
    
    
    
    user_audio_encoder = UserPhonemeEncoder(max_phonemes=audio_len, uncertianty_vector_size=uncertainty_vecctor_size, dense_width=64, dense1_width=128, dropout=0.1)
    target_phoneme_encoder = TargetPhonemeEncoder(vocab_size=vocab_len, max_phonemes=target_len, embeded_vector_size=128, dense_width=64, dense1_width=128, dropout=0.1)
    cross_attention_transformer = CrossAttentionTransformer(embeded_vector_size=128, uncertainty_vector_size=uncertainty_vecctor_size, dense=64, dense1=128, dropout=0.1)
    classifier_head = ClassifierHead(embeded_vector_size=128, dense1=64, dense2=128, dropout=0.1)
    print("point 6")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    print("point 7")
    val_loss = train(train_dataset=train_dataset_batched, val_dataset=val_dataset_batched, user_encoder=user_audio_encoder, target_encoder=target_phoneme_encoder, transformer=cross_attention_transformer, classifier=classifier_head, optimizer=optimizer, loss_fn=loss_fn)
    
    print(f"val loss: {val_loss}")
    
    
    out_path = Path(__file__).resolve().parents[0] / "fluency_trainer_model"
    
    if out_path.is_dir():
        shutil.rmtree(out_path)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    user_audio_encoder.save_weights(out_path / "user_encoder.weights.h5")
    user_audio_encoder.save_weights(out_path / "target_encoder.weights.h5")
    user_audio_encoder.save_weights(out_path / "cross_attention_transformer.weights.h5")
    user_audio_encoder.save_weights(out_path / "classifier_head.weights.h5")
    
    api = HfApi()

    if get_token() != None:
        api.upload_folder(
            folder_path=out_path,
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="model",
            path_in_repo="speech2target",
            delete_patterns="*"   
        )
    
if __name__ == "__main__":
    build_model()