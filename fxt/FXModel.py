import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

class ProbabilityDistribution(keras.Model):
    def call(self, inputs):
        return tf.squeeze(tf.random.categorical(inputs, 1), axis=-1)

class FXModel(keras.Model):
    def __init__(self, action_size):
        super().__init__('mlp_policy')

        self.lstm1 = LSTM(256, return_sequences=True)
        self.lstm2 = LSTM(256, return_sequences=True)
        self.lstm3 = LSTM(128, return_sequences=True)
        self.lstm4 = LSTM(128)
        self.h = Dense(64, activation='relu')
        
        self.logits = Dense(action_size, activation='softmax')
        self.value = Dense(1, activation='linear')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        c = self.lstm1(x)
        c = self.lstm2(c)
        c = self.lstm3(c)
        c = self.lstm4(c)
        c = self.h(c)

        return self.logits(c), self.value(c)

    def action_value(self, inputs):
        logits, value = self.predict_on_batch(inputs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

from env.FXEnv import FXEnv

env = FXEnv()
obs = env.reset()

model = FXModel(6)
model.predict(obs.reshape(1, obs.shape[0], obs.shape[1]))