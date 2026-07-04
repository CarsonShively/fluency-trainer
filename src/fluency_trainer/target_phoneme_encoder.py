import tensorflow as tf

class TargetPhonemeEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embeded_vector_size, max_phonemes, dense_width, dense1_width):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embeded_vector_size = embeded_vector_size
        self.max_phonemes = max_phonemes
        self.dense_width = dense_width
        
        lookup_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05) 
        transform_initializer = tf.keras.initializers.GlorotUniform()
        bias1_initialier = tf.zeros(dense1_width)
        bias2_initialier = tf.zeros(embeded_vector_size)

        self.ids_vector = tf.zeros(max_phonemes)
        
        self.phoneme_embedding_map = tf.Variable(
            lookup_initializer(shape=[vocab_size, embeded_vector_size]),
            trainable=True 
        )
        
        self.phoneme_embedding_map.assign(
            tf.tensor_scatter_nd_update(
                self.phoneme_embedding_map,
                indices=[[0]],
                updates=[tf.zeros(embeded_vector_size)]
            )
        )
        
        self.position_map = tf.Variable(
            lookup_initializer(shape=[max_phonemes, embeded_vector_size]),
            trainable=True
        )
        
        self.q = tf.Variable(
            transform_initializer(shape=[embeded_vector_size, dense_width]),
            trainable=True
        )
        
        self.k = tf.Variable(
            transform_initializer(shape=[embeded_vector_size, dense_width]),
            trainable=True
        )

        self.v = tf.Variable(
            transform_initializer(shape=[embeded_vector_size, dense_width]),
            trainable=True
        )        
        
        self.project_attention = tf.Variable(
            transform_initializer(shape=[dense_width, embeded_vector_size]),
            trainable=True
        )
        
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        
        self.dense1 = tf.Variable(
            transform_initializer(shape=[embeded_vector_size, dense1_width]),
            trainable=True
        )
        
        self.bias1 = tf.Variable(
            bias1_initialier,
            trainable=True
        )
        
        self.dense2 = tf.Variable(
            transform_initializer(shape=[dense1_width, embeded_vector_size]),
            trainable=True
        )
        
        self.bias2 = tf.Variable(
            bias2_initialier,
            trainable=True
        )
        
        self.layer_norm2 = tf.keras.layers.LayerNormalization()



    def call(self, x):
        
        if not tf.is_tensor(x):
            raise ValueError("x not tensor")
        
        if x.shape[1] > self.max_phonemes:
            raise ValueError("x to long")
        
        padded_ids = tf.pad(x, paddings=[[0, 0], [0, self.max_phonemes - x.shape[1]]], constant_values=0)
        valid_positions = tf.not_equal(padded_ids, 0)
        context_mask = valid_positions[:, tf.newaxis, :]
        output_mask = valid_positions[:, :, tf.newaxis]
        
        embeded_matrix = self.phoneme_embedding_map[padded_ids]
        
        position_aware_matrix = embeded_matrix + self.position_map
        
        q_matrix = position_aware_matrix @ self.q
        k_matrix = position_aware_matrix @ self.k
        v_matrix = position_aware_matrix @ self.v
        
        transposed_k_matrix = tf.transpose(k_matrix, perm=[0, 2, 1])
        
        context_matrix = (q_matrix @ transposed_k_matrix) / tf.sqrt(tf.cast(self.dense_width, tf.float32))
        
        context_matrix_masked = tf.where(
            context_mask,
            context_matrix,
            tf.cast(-1e9, context_matrix.dtype)
        )
        
        softmax_context_matrix = tf.nn.softmax(context_matrix_masked, axis=-1)
        
        attention_matrix = softmax_context_matrix @ v_matrix
        
        projected_attention_matrix = attention_matrix @ self.project_attention

        residual_matrix = projected_attention_matrix + position_aware_matrix
        
        residual_matrix_norm = self.layer_norm1(residual_matrix)
        
        layer1 = residual_matrix_norm @ self.dense1 + self.bias1
        
        layer1_activation = tf.keras.activations.gelu(layer1)
        
        layer2 = layer1_activation @ self.dense2 + self.bias2
        
        residual2_matrix = residual_matrix_norm + layer2
        
        residual2_matrix_norm = self.layer_norm2(residual2_matrix)
        
        target_self_attention = residual2_matrix_norm * tf.cast(output_mask, residual2_matrix_norm.dtype)
        
        return target_self_attention, valid_positions