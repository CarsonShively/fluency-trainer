from huggingface_hub import snapshot_download, HfApi, get_token
from pathlib import Path
import tensorflow as tf
from fluency_trainer.phone_audio_alignment_model import PhoneAudioAlignmentModel
import json
import numpy as np


def evaluate_model():
    
    allow_patterns = [
        "matrices/test_user_audio.npy",
        "matrices/test_user_audio_mask.npy",
        "matrices/test_target_phonemes.npy",
        "matrices/test_target_phonemes_mask.npy",
        "matrices/test_target_classes.npy",
        "vocab/**"
    ]
    
    local_data = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns=allow_patterns
    ))
    
    with open(local_data / "vocab/vocab.json", "r") as con:
        vocab = json.load(con)
    
    vocab_len = len(vocab)
    
    local_model = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="model",
        allow_patterns="phone_audio_alignment_model.weights.h5"
    ))
    
    model = PhoneAudioAlignmentModel(vocab_size=vocab_len)
    
    
    
    audio = np.load(local_data / "matrices/test_user_audio.npy")
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    del audio
    
    audio_mask = np.load(local_data / "matrices/test_user_audio_mask.npy")
    audio_mask_tensor = tf.convert_to_tensor(audio_mask, dtype=tf.float32)
    del audio_mask
    
    phones = np.load(local_data / "matrices/test_target_phonemes.npy")
    phones_tensor = tf.convert_to_tensor(phones, dtype=tf.int32)
    del phones
    
    phones_mask = np.load(local_data / "matrices/test_target_phonemes_mask.npy")
    phones_mask_tensor = tf.convert_to_tensor(phones_mask, dtype=tf.float32)
    del phones_mask
    
    scores = np.load(local_data / "matrices/test_target_classes.npy")
    scores_tensor = tf.convert_to_tensor(scores, dtype=tf.float32)
    del scores
    
    model(audio=audio_tensor[0:1], phones=phones_tensor[0:1], audio_mask=audio_mask_tensor[0:1], phones_mask=phones_mask_tensor[0:1])
    model.load_weights(local_model / "phone_audio_alignment_model.weights.h5")
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_audio": audio_tensor,
            "user_audio_mask": audio_mask_tensor,
            "target_phones": phones_tensor,
            "target_phones_mask": phones_mask_tensor
        },
        scores_tensor
    )).batch(16)
    
    loss_fn = tf.losses.MeanSquaredError()
    
    loss_sum = 0
    batch_sum = 0
    
    for batch, labels in test_dataset:
        model_out = model(audio=batch["user_audio"], audio_mask=batch["user_audio_mask"], phones=batch["target_phones"], phones_mask=batch["target_phones_mask"])
    
        a_loss = loss_fn(model_out["accuracy"], labels[:, 0:1])
        f_loss = loss_fn(model_out["fluency"], labels[:, 1:2])
        p_loss = loss_fn(model_out["prosodic"], labels[:, 2:3])
        t_loss = loss_fn(model_out["total"], labels[:, 3:4])
    
        loss = a_loss + f_loss + p_loss + t_loss
        loss_sum += loss
        batch_sum += 1
    
    avg_loss = loss_sum / batch_sum
    
    holdout_loss = float(avg_loss.numpy())
    
    print(f"holdout loss: {holdout_loss}")
    
    report = {
        "holdout_mse": holdout_loss
    }
    
    out_path = Path(__file__).resolve().parents[0] / "evaluation_report.json"
    
    out_path.unlink(missing_ok=True)
    
    with open(out_path, "w") as con:
        json.dump(report, con)
    
    if get_token() != None:
        api = HfApi()
        api.upload_file(
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="model",
            path_or_fileobj=out_path,
            path_in_repo="evaluation_report.json",
        )
    
if __name__ == "__main__":
    evaluate_model()
