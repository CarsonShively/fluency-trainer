from huggingface_hub import snapshot_download, HfApi, get_token
from pathlib import Path
import tensorflow as tf
import numpy as np
import json

def mean_baseline():
    
    local_data = Path(snapshot_download(
        repo_id="Carson-Shively/fluency-trainer",
        repo_type="dataset",
        allow_patterns=["matrices/train_target_classes.npy", "matrices/val_target_classes.npy"]
    ))
    
    train_scores = np.load(local_data / "matrices/train_target_classes.npy")
    val_scores = np.load(local_data / "matrices/val_target_classes.npy")
    
    loss_fn = tf.losses.MeanSquaredError()
    
    val_mean = np.zeros(shape=val_scores.shape)
    
    val_mean[:, 0] = np.mean(train_scores[:, 0])
    val_mean[:, 1] = np.mean(train_scores[:, 1])
    val_mean[:, 2] = np.mean(train_scores[:, 2])
    val_mean[:, 3] = np.mean(train_scores[:, 3])
    
    
    accuracy_loss = loss_fn(val_mean[:, 0], val_scores[:, 0])
    fluency_loss = loss_fn(val_mean[:, 1], val_scores[:, 1])
    ps_loss = loss_fn(val_mean[:, 2], val_scores[:, 2])
    total_loss = loss_fn(val_mean[:, 3], val_scores[:, 3])
    
    loss = accuracy_loss + fluency_loss + ps_loss + total_loss
    
    mean_loss = float(loss.numpy())
    
    print(f"mean baseline loss: {mean_loss}")
    
    report = {
        "mean_baseline_mse": mean_loss
    }
    
    out_path = Path(__file__).resolve().parents[0] / "mean_baseline_benchmark.json"
    
    with open(out_path, "w") as con:
        json.dump(report, con)
    
    if get_token() != None:
        api = HfApi()
        
        api.upload_file(
            repo_id="Carson-Shively/fluency-trainer",
            repo_type="model",
            path_or_fileobj=out_path,
            path_in_repo="mean_baseline_benchmark.json",
        )
    
if __name__ == "__main__":
    mean_baseline()