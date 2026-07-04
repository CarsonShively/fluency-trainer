import tensorflow as tf

class CrossAttentionTransformer(tf.keras.Model):
    def __init__(self, embeded_vector_size, dense, dense1):
        super().__init__()
        
        self.embeded_vector_size = embeded_vector_size
        self.dense = dense
        self.dense1 = dense1
        
        transform_initialier = tf.keras.initializers.GlorotUniform()
        
        self.q = tf.Variable(
            transform_initialier(shape=[embeded_vector_size, dense]),
            trainable=True
        )
        
        self.k = tf.Variable(
            transform_initialier(shape=[embeded_vector_size, dense]),
            trainable=True
        )

        self.v = tf.Variable(
            transform_initialier(shape=[embeded_vector_size, dense]),
            trainable=True
        )
        
        self.project_context_attention = tf.Variable(
            transform_initialier(shape=[dense, embeded_vector_size]),
            trainable=True
        )
        
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        
        self.layer1 = tf.Variable(
            transform_initialier(shape=[embeded_vector_size, dense1]),
            trainable=True
        )
        
        self.bias1 = tf.Variable(
            tf.zeros(dense1),
            trainable=True
        )
        
        self.layer2 = tf.Variable(
            transform_initialier(shape=[dense1, embeded_vector_size]),
            trainable=True
        )
        
        self.bias2 = tf.Variable(
            tf.zeros(embeded_vector_size),
            trainable=True
        )
        
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        

    
    def call(self, target_attention, user_attention, target_valid_positions, user_valid_positions):
        
        if not tf.is_tensor(target_attention) or not tf.is_tensor(user_attention):
            raise TypeError("an attention matrix is not a tensor.")
        
        user_key_mask = user_valid_positions[:, tf.newaxis, :]
        output_mask = target_valid_positions[:, :, tf.newaxis]
        
        
        target_query = target_attention @ self.q

        user_key = user_attention @ self.k
        user_value = user_attention @ self.v
        
        user_key_transposed = tf.transpose(user_key, perm=[0, 2, 1])
        
        context = (target_query @ user_key_transposed) / tf.cast(tf.sqrt(self.dense), target_query.dtype)
        
        context_masked = tf.where(
            user_key_mask,
            context,
            tf.cast(-1e9, context.dtype)
        )
        
        softmax_context = tf.nn.softmax(context_masked, axis=-1)
        
        cross_attention = softmax_context @ user_value
        
        projected_context_attention = cross_attention @ self.project_context_attention
        
        residual = projected_context_attention + target_attention
        
        residual_norm = self.layer_norm1(residual)
        
        layer1 = residual_norm @ self.layer1 + self.bias1
        
        layer1_activation = tf.keras.activations.gelu(layer1)
        
        layer2 = layer1_activation @ self.layer2 + self.bias2
        
        residual2 = residual_norm + layer2
        
        cross_attention = self.layer_norm2(residual2)
        
        cross_attention_output = cross_attention * tf.cast(output_mask, cross_attention.dtype)
        
        return cross_attention_output
    
    
