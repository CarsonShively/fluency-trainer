import tensorflow as tf

# do it myself but do it quickly
class PhoneAudioAlignmentModel(tf.keras.Model):
    def __init__(self, max_phones_len, phone_vector_size, projection1):
        super().__init__()
        self.max_phones_len = max_phones_len
        
        self.phone_embeding_map = tf.keras.layers.Embedding(input_dim=max_phones_len, output_dim=phone_vector_size)
        self.phone_projection1 = tf.keras.layers.Dense(projection1)
        self.audio_projection1 = tf.keras.layers.Dense(projection1)

    def call(self, phones, audio, phones_mask, audio_mask):
        
        embeded_phones = self.phone_embeding_map(phones)
        
        projected_phones = self.phone_projection1(embeded_phones)
        projected_audio = self.audio_projection1(audio)
        
        alignment_matrix = tf.matmul(
            projected_phones,
            projected_audio,
            transpose_b=True
        )