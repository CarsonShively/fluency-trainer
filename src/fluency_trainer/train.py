import tensorflow as tf

def train(train_dataset, val_dataset, model, optimizer, loss_fn):
    max_epochs = 100
    patience = 10
    patience_counter = 0
    best_loss = float("inf")
    epsilon = 0.001
    
    for epoch in range(max_epochs):
        train_loss_sum = 0
        train_batches = 0
        for batch, labels in train_dataset:
            
            with tf.GradientTape() as tape:
                
                output = model(audio=batch["user_audio"], phones=batch["target_phones"], audio_mask=batch["user_audio_mask"], phones_mask=batch["target_phones_mask"])
                
                accuracy_loss = loss_fn(output["accuracy"], labels[:, 0:1])
                fluency_loss = loss_fn(output["fluency"], labels[:, 1:2])
                prosodic_loss = loss_fn(output["prosodic"], labels[:, 2:3])
                total_loss = loss_fn(output["total"], labels[:, 3:4])
                
                loss = accuracy_loss + fluency_loss + prosodic_loss + total_loss
                
                
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss_sum += loss
            train_batches += 1
        
        train_loss = train_loss_sum / train_batches
        
        print(f"epoch {epoch + 1} train loss: {train_loss}")
        
        val_loss_sum = 0
        batch_counter = 0
            
        for batch, labels in val_dataset:
            
            output = model(audio=batch["user_audio"], phones=batch["target_phonemes"], audio_mask=batch["user_audio_mask"], phones_mask=batch["target_phonemes_mask"])
            
            accuracy_loss = loss_fn(output["accuracy"], labels[:, 0:1])
            fluency_loss = loss_fn(output["fluency"], labels[:, 1:2])
            prosodic_loss = loss_fn(output["prosodic"], labels[:, 2:3])
            total_loss = loss_fn(output["total"], labels[:, 3:4])
            
            loss = accuracy_loss + fluency_loss + prosodic_loss + total_loss
            
            val_loss_sum += loss
            batch_counter += 1
            
        avg_loss = val_loss_sum / batch_counter
        
        print(f"epoch {epoch + 1} val loss: {avg_loss}")
        
        if avg_loss + epsilon < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            return best_loss
        
    return best_loss