from huggingface_hub import snapshot_download
from pathlib import Path
import tensorflow as tf
import numpy as np

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
    
    print(f"mean baseline loss: {loss}")
    
if __name__ == "__main__":
    mean_baseline()