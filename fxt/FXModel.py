import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM

class FXModel(keras.Model):
    def __init__(self, action_size):
        super().__init__('mlp_policy')

        self.lstm1 = LSTM(50, return_sequences=True)
        self.lstm2 = LSTM(50, return_sequences=True)
        self.lstm3 = LSTM(50, return_sequences=True)
        self.lstm4 = LSTM(50)

    
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        c = self.lstm1(x)
        c = self.lstm2(x)
        c = self.lstm3(x)
        c = self.lstm4(x)

        print(c)
        return c

from env.FXEnv import FXEnv

env = FXEnv()
obs = env.reset()

model = FXModel(6)
model.predict(obs.reshape(1, obs.shape[0], obs.shape[1]))