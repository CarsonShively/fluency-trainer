import tensorflow as tf

def train(dataset, user_encoder, target_encoder, transformer, classifier, optimizer, loss_fn):
    max_epochs = 100
    
    for epoch in range(max_epochs):
        for batch, labels, labels_mask in dataset:
            
            with tf.GradientTape() as tape:
                user_attention = user_encoder(batch["user_audio"], batch["user_audio_mask"])
                target_attention = target_encoder(batch["target_phones"], batch["target_phones_mask"])
                
                cross_attention = transformer(user_attention, target_attention, batch["user_audio_mask"], batch["target_phones_mask"])
                
                logits = classifier(cross_attention)
                
                loss = loss_fn(labels, logits)
                
                classes_mask = tf.cast(labels_mask, tf.float32)
                
                loss_masked = loss * classes_mask
                
                loss_avg = tf.reduce_sum(loss_masked) / tf.reduce_sum(labels_mask)
                
            all_variables = user_encoder.trainable_variables + target_encoder.trainable_variables + transformer.trainable_variables + classifier.trainable_variables
            
            gradients = tape.gradient(loss, all_variables)
            optimizer.apply_gradients(zip(gradients, all_variables))