import tensorflow as tf

class ClassifierHead(tf.keras.Model):
    def __init__(self, embeded_vector_size, dense1, dense2, dropout):
        super().__init__()
        
        self.dropout = dropout
        
        dense_init = tf.keras.initializers.GlorotUniform()
        
        self.layer1 = tf.Variable(
            dense_init(shape=[embeded_vector_size, dense1]),
            trainable=True
        )
        
        self.bias1 = tf.Variable(
            tf.zeros(dense1),
            trainable=True
        )

        self.layer2 = tf.Variable(
            dense_init(shape=[dense1, dense2]),
            trainable=True
        )
        
        self.bias2 = tf.Variable(
            tf.zeros(dense2),
            trainable=True
        )
        
        self.layer3 = tf.Variable(
            dense_init(shape=[dense2, 3]),
            trainable=True
        )

        self.bias3 = tf.Variable(
            tf.zeros(3),
            trainable=True
        )
        
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        
    def call(self, cross_attention, training):
        
        layer1_out = cross_attention @ self.layer1 + self.bias1
        layer1_out_activated = tf.keras.activations.relu(layer1_out)
        layer1_out_activated = self.dropout_layer(layer1_out_activated, training=training)
        layer2_out = layer1_out_activated @ self.layer2 + self.bias2
        layer2_out_activated = tf.keras.activations.relu(layer2_out)
        layer2_out_activated = self.dropout_layer(layer2_out_activated, training=training)
        layer3_out = layer2_out_activated @ self.layer3 + self.bias3
        
        return layer3_out