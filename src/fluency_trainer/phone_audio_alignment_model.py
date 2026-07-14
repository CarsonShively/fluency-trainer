import tensorflow as tf

class PhoneAudioAlignmentModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.phone_vector_size = 64
        self.projection1 = 128
        self.filters = 32
        self.filter_y = 3
        self.filter_x = 5
        self.lstm_units = 64
        
        self.phone_embeding_map = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.phone_vector_size)
        self.phone_projection1 = tf.keras.layers.Dense(self.projection1)
        self.audio_projection1 = tf.keras.layers.Dense(self.projection1)
        
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(self.filter_y, self.filter_x), strides=(1,1), padding="same", activation=None)
        self.norm1 = tf.keras.layers.LayerNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(self.filter_y, self.filter_x), strides=(1,1), padding="same", activation=None)
        self.norm2 = tf.keras.layers.LayerNormalization()
        
        self.conv3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(self.filter_y, self.filter_x), strides=(1,1), padding="same", activation=None)
        self.norm3 = tf.keras.layers.LayerNormalization()

        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=self.lstm_units,
                return_sequences=True
            )
        )
        
        self.hidden1 = tf.keras.layers.Dense(64)
        self.accuracy = tf.keras.layers.Dense(1)
        self.fluency = tf.keras.layers.Dense(1)
        self.prosodic = tf.keras.layers.Dense(1)
        
        self.final_hidden = tf.keras.layers.Dense(32)
        self.total = tf.keras.layers.Dense(1)

    def call(self, phones, audio, phones_mask, audio_mask):
        
        embeded_phones = self.phone_embeding_map(phones)
        
        projected_phones = self.phone_projection1(embeded_phones)
        projected_audio = self.audio_projection1(audio)
        
        alignment_matrix = tf.matmul(
            projected_phones,
            projected_audio,
            transpose_b=True
        )
        
        scaled_alignment_matrix = alignment_matrix / (tf.math.sqrt(tf.cast(self.projection1, dtype=alignment_matrix.dtype)))
        
        phones_mask_2d = phones_mask[:, :, None]
        audio_mask_2d = audio_mask[:, None, :]
        
        alignment_matrix_mask = tf.cast((phones_mask_2d * audio_mask_2d), dtype=scaled_alignment_matrix.dtype)
        
        masked_alignment = scaled_alignment_matrix * alignment_matrix_mask
        
        cnn_alignment_input = masked_alignment[:, :, :, None]
        
        conv1_output = self.conv1(cnn_alignment_input)
        conv1_norm = self.norm1(conv1_output)
        residual = tf.keras.activations.gelu(conv1_norm)
        
        
        
        conv2_output = self.conv2(residual)
        conv2_norm = self.norm2(conv2_output)
        conv2_activation = tf.keras.activations.gelu(conv2_norm)
        
        residual2 = residual + conv2_activation
        
        conv3_output = self.conv3(residual2)
        conv3_norm = self.norm3(conv3_output)
        conv3_activation = tf.keras.activations.gelu(conv3_norm)
        
        residual3 = residual2 + conv3_activation
        
        alignment_matrix_mask2 = alignment_matrix_mask[:, :, :, None]
        
        residual_final = residual3 * alignment_matrix_mask2
        
        residual_final_sum = tf.reduce_sum(residual_final, axis=2)
        
        valid_positions = tf.maximum(tf.reduce_sum(alignment_matrix_mask2, axis=2), 1.0)
        
        audio_compression_matrix = residual_final_sum / valid_positions
        
        bilstm_matrix = self.bilstm(audio_compression_matrix, mask=tf.cast(phones_mask, dtype=tf.bool))
        
        bilstm_phones_sums = tf.reduce_sum((bilstm_matrix * tf.cast(phones_mask_2d, dtype=bilstm_matrix.dtype)), axis=1)
        bilsm_valid_positions = tf.maximum(tf.reduce_sum(phones_mask_2d, axis=1), 1.0)
        
        alignment_vector = bilstm_phones_sums / tf.cast(bilsm_valid_positions, dtype=bilstm_phones_sums.dtype)
        
        hidden1_out = self.hidden1(alignment_vector)
        hidden1_out_act = tf.keras.activations.gelu(hidden1_out)
        
        accuracy_out = self.accuracy(hidden1_out_act)
        fluency_out = self.fluency(hidden1_out_act)
        prosodic_out = self.prosodic(hidden1_out_act)
        
        final_input = tf.concat([accuracy_out, fluency_out, prosodic_out], axis=-1)
        
        final_hidden_out = self.final_hidden(final_input)
        final_hidden_out_act = tf.keras.activations.gelu(final_hidden_out)
        total_out = self.total(final_hidden_out_act)
        
        out_vals = {
            "accuracy": accuracy_out, 
            "fluency": fluency_out, 
            "prosodic":prosodic_out, 
            "total": total_out
        }
        
        return out_vals