import tensorflow as tf

class UserPhonemeEncoder(tf.keras.Model):
    def __init__(self, max_phonemes, uncertianty_vector_size, dense_width, dense1_width, dropout):
        super().__init__()
        
        self.uncertianty_vector_size = uncertianty_vector_size
        self.max_phonemes = max_phonemes
        self.dense_width = dense_width
        self.dropout = dropout
        
        lookup_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) 
        transform_initializer = tf.keras.initializers.GlorotUniform()
        bias1_initialier = tf.zeros(dense1_width)
        bias2_initialier = tf.zeros(uncertianty_vector_size)
        
        self.position_map = tf.Variable(
            lookup_initializer(shape=[max_phonemes, uncertianty_vector_size]),
            trainable=True
        )
        
        self.q = tf.Variable(
            transform_initializer(shape=[uncertianty_vector_size, dense_width]),
            trainable=True
        )
        
        self.k = tf.Variable(
            transform_initializer(shape=[uncertianty_vector_size, dense_width]),
            trainable=True
        )

        self.v = tf.Variable(
            transform_initializer(shape=[uncertianty_vector_size, dense_width]),
            trainable=True
        )        
        
        self.project_attention = tf.Variable(
            transform_initializer(shape=[dense_width, uncertianty_vector_size]),
            trainable=True
        )
        
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        
        self.dense1 = tf.Variable(
            transform_initializer(shape=[uncertianty_vector_size, dense1_width]),
            trainable=True
        )
        
        self.bias1 = tf.Variable(
            bias1_initialier,
            trainable=True
        )
        
        self.dense2 = tf.Variable(
            transform_initializer(shape=[dense1_width, uncertianty_vector_size]),
            trainable=True
        )
        
        self.bias2 = tf.Variable(
            bias2_initialier,
            trainable=True
        )
        
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        self.dropout_layer = tf.keras.layers.Dropout(dropout)


    def call(self, x, mask, training):
        
        if not tf.is_tensor(x):
            raise ValueError("x not tensor")
        
        context_mask = mask[:, tf.newaxis, :]
        context_mask = tf.cast(context_mask, dtype=tf.bool)
        output_mask = mask[:, :, tf.newaxis]
        
        batch_sequence_len = tf.shape(x)[1]
        
        position_aware_matrix = x + self.position_map[:batch_sequence_len]
        
        q_matrix = position_aware_matrix @ self.q
        k_matrix = position_aware_matrix @ self.k
        v_matrix = position_aware_matrix @ self.v
        
        transposed_k_matrix = tf.transpose(k_matrix, perm=[0, 2, 1])
        
        context_matrix = (q_matrix @ transposed_k_matrix) / tf.sqrt(tf.cast(self.dense_width, dtype=q_matrix.dtype))
        
        context_matrix_masked = tf.where(
            context_mask,
            context_matrix,
            tf.cast(-1e9, context_matrix.dtype)
        )
        
        softmax_context_matrix = tf.nn.softmax(context_matrix_masked, axis=-1)
        
        attention_matrix = softmax_context_matrix @ v_matrix
        
        projected_attention_matrix = attention_matrix @ self.project_attention

        residual_matrix = self.dropout_layer(projected_attention_matrix, training=training) + position_aware_matrix
        
        residual_matrix_norm = self.layer_norm1(residual_matrix)
        
        layer1 = residual_matrix_norm @ self.dense1 + self.bias1
        
        layer1_activation = tf.keras.activations.gelu(layer1)
        
        layer2 = layer1_activation @ self.dense2 + self.bias2
        
        residual2_matrix = residual_matrix_norm + self.dropout_layer(layer2, training=training)
        
        residual2_matrix_norm = self.layer_norm2(residual2_matrix)
        
        target_self_attention = residual2_matrix_norm * tf.cast(output_mask, residual2_matrix_norm.dtype)
        
        return target_self_attention