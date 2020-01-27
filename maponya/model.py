import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

class ProbabilityDistribution(keras.Model):
    def call(self, inputs):
        return tf.squeeze(tf.random.categorical(inputs, 1), axis=-1)


class MyModel(keras.Model):
    def __init__(self, action_size):
        super().__init__('mlp_policy')

        self.h1 = Dense(16, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.h3 = Dense(128, activation='relu')
        self.h4 = LSTM(128)
        
        self.logits = Dense(action_size, activation='softmax')
        self.value = Dense(1, activation='linear')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        
        h = self.h1(x)
        h = self.h2(h)
        h = self.h3(h)
        h = self.h4(h)

        return self.logits(h), self.value(h)

    def action_value(self, inputs):
        logits, value = self.predict_on_batch(inputs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)