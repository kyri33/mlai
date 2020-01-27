import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import numpy as np

class MyAutoencoder(Model):
    def __init__(self):
        super().__init__(MyAutoencoder)
        input_size = 12
        hidden_size = 10
        code_size = 16
        
        self.hidden1 = Dense(hidden_size, activation='relu')
        self.code = Dense(code_size, activation='relu')
        self.hidden2 = Dense(hidden_size, activation='relu')
        self.os = Dense(input_size, activation='tanh')
    
    def encode(self, inputs):
        x = tf.convert_to_tensor(inputs)
        h = self.hidden1(x)
        return self.code(h)

    def decode(self, codes):
        h = self.hidden2(codes)
        return self.os(h)
    
    def call(self, inputs):
        code = self.encode(inputs)
        return self.decode(code)

    def noise_and_fit(self, inputs):
        m = 0.0
        std = 0.1
        noise = np.random.normal(m, std, inputs.shape)
        x_noise = inputs + noise
        self.fit(x_noise, inputs, batch_size=1, epochs=1)