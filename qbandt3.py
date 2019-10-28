import tensorflow as tf
from tensorflow import keras
import numpy as np

class Bandits:
    def __init__(self):
        self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.state = 0
        self.action_size = self.bandits.shape[1]

    def pickBandit(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state
    
    def pullBandit(self, state, action):
        band = self.bandits[state, action]
        ra = np.random.randn(1)
        if ra > band:
            return 1.0
        else:
            return -1.0
        

model = keras.Sequential()
bandos = Bandits()

model.add(keras.Dense(4, input_shape=(bandos.num_bandits,)))