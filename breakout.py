import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray
import collections
from tensorflow import keras
import gym

env = gym.make("Breakout-v0")

action_size = env.action_space.n

def process_frame(frame):
    gray = rgb2gray(frame)
    normalised_frame = gray/255.
    processed_frame = transform.resize(normalised_frame, [110, 84])
    return processed_frame

stack_size = 4
state_size = [110, 84, stack_size]
stacked_frames = collections.deque(maxlen=stack_size)
def stack_frames(state, is_new=False):
    global stacked_frames
    global stack_size
    state = process_frame(state)
    if is_new:
        for _ in range(stack_size):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)
    return np.stack(stacked_frames, axis=2)
    

class Memory():
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen = max_size)
    
    def add(self, state):
        self.buffer.append(state)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indexes = np.random.choice(np.arange(buffer_size), 
            size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]

memory_size = 100000
memory = Memory(memory_size)
batch_size = 64
for i in range(batch_size):
    if i == 0:
        state = env.reset()
        state = stack_frames(state, is_new = True)
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    if done:
        next_state = np.zeros(state.shape)
        next_state = stack_frames(next_state)
        memory.add((state, action, reward, next_state, done))
    else:
        next_state = stack_frames(next_state)
        memory.add((state, action, reward, next_state, done))
        state = next_state

learning_rate = 0.00025

inputs = keras.Input(shape=tuple(state_size))
d = keras.layers.Conv2D(32, 8, 
        strides=4, activation='elu')(inputs)
d = keras.layers.Conv2D(64, 4, strides=2, 
        activation="elu")(d)
d = keras.layers.Flatten()(d)
d = keras.layers.Dense(512, activation='elu')(d)
output = keras.layers.Dense(action_size, activation="linear")(d)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

decay_rate = 0.00001
gamma = 0.9

def pick_action(state, decay_step):
    e = np.random.rand(1)[0]
    #epsilon = np.exp(-decay_rate * decay_step)
    epsilon = 0.0
    if epsilon > e:
        action = env.action_space.sample()
    else:
        action = np.argmax(model.predict(state))
    return action, epsilon

n_episodes = 50000
max_steps = 50000
episode_render = True

decay_step = 1
reward_list = []
for episode in range(n_episodes):
    state = env.reset()
    state = stack_frames(state, is_new=True)
    step = 0
    loss = 0.0000
    episode_rewards = []
    
    while step < max_steps:
        step += 1
        action, epsilon = pick_action(state, decay_step)
        decay_step += 1
        next_state, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        if episode_render:
            env.render()
        if done:
            next_state = np.zeros(state.shape)
            next_state = stack_frames(next_state)
            memory.add((state, action, reward, next_state, done))
            reward_list.append((episode, np.sum(episode_rewards)))
            break
        else:
            next_state = stack_frames(next_state)
            memory.add((state, action, reward, next_state, done))
            state = next_state
        
    