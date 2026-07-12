import tensorflow as tf

def train(train_dataset, val_dataset, user_encoder, target_encoder, transformer, classifier, optimizer, loss_fn):
    max_epochs = 100
    patience = 10
    patience_counter = 0
    best_loss = float("inf")
    epsilon = 0.001
    
    for epoch in range(max_epochs):
        for batch, labels, labels_mask in train_dataset:
            
            with tf.GradientTape() as tape:
                user_attention = user_encoder(x=batch["user_audio"], padding_mask=batch["user_audio_mask"], training=True)
                target_attention = target_encoder(x=batch["target_phones"], padding_mask=batch["target_phones_mask"], training=True)
                
                cross_attention = transformer(user_attention=user_attention, target_attention=target_attention, user_mask=batch["user_audio_mask"], target_mask=batch["target_phones_mask"], training=True)
                
                logits = classifier(cross_attention, training=True)
                
                loss = loss_fn(labels, logits)
                
                classes_mask = tf.cast(labels_mask, tf.float32)
                
                loss_masked = loss * classes_mask
                
                loss_avg = tf.reduce_sum(loss_masked) / tf.reduce_sum(tf.cast(labels_mask, dtype=loss_masked.dtype))
                
            all_variables = user_encoder.trainable_variables + target_encoder.trainable_variables + transformer.trainable_variables + classifier.trainable_variables
            
            gradients = tape.gradient(loss_avg, all_variables)
            optimizer.apply_gradients(zip(gradients, all_variables))
            
        total_loss = 0
        batch_counter = 0
            
        for batch, labels, labels_mask in val_dataset:
            
            user_attention = user_encoder(x=batch["user_audio"], padding_mask=batch["user_audio_mask"], training=False)
            target_attention = target_encoder(x=batch["target_phones"], padding_mask=batch["target_phones_mask"], training=False)
            
            cross_attention = transformer(user_attention=user_attention, target_attention=target_attention, user_mask=batch["user_audio_mask"], target_mask=batch["target_phones_mask"], training=False)
            
            logits = classifier(cross_attention, training=False)
            
            loss = loss_fn(labels, logits)
            
            classes_mask = tf.cast(labels_mask, tf.float32)
            
            loss_masked = loss * classes_mask
            
            loss_avg = tf.reduce_sum(loss_masked) / tf.reduce_sum(tf.cast(labels_mask, dtype=loss_masked.dtype))
            
            total_loss += loss_avg
            
            batch_counter += 1
            
        avg_loss = total_loss / batch_counter
        
        if avg_loss + epsilon < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter == patience:
            return best_loss
        
    return best_loss