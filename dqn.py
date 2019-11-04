import tensorflow as tf
import numpy as np
import collections
from tensorflow import keras
import random
import gym
#from tensorflow.keras import layers, Sequential

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=(self.state_size,), activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer = keras.optimizers.Adam(lr = self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        rand = np.random.rand(1)
        if rand < self.epsilon:
            return np.random.randint(0, self.action_size), True
        act = self.model.predict(state)
        return np.argmax(act[0]), False

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        
episodes = 1000

env = gym.make("CartPole-v0")
agent = DQNAgent(4, 2)
totalScore = 0
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])

    for tim in range(500):
        action, ra = agent.act(state)
        print(action, ra)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            totalScore += tim
            print("episode: {}/{}, score: {}".format(e, episodes, tim))
            break

    batch_size = 32
    if totalScore < batch_size:
        batch_size = totalScore
    agent.replay(batch_size)