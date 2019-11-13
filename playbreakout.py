import tensorflow as tf
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
import collections
from tensorflow import keras
import gym
from skimage import io
from matplotlib import pyplot as plt

env = gym.make("BreakoutDeterministic-v4")

action_size = env.action_space.n

def process_frame(frame):
    gray = np.mean(frame, axis=2).astype(np.uint8)
    gray = gray[::2, ::2]
    return gray

stack_size = 4
state_size = [105, 80, stack_size]
stacked_frames = collections.deque(maxlen=stack_size)
def stack_frames(state, is_new=False):
    #global stacked_frames
    #global stack_size
    state = process_frame(state)
    if is_new:
        for _ in range(stack_size):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)
    return np.stack(stacked_frames, axis=2)

learning_rate = 0.00025

inputs = keras.Input(shape=tuple(state_size))
norm = keras.layers.Lambda(lambda x: x/255.0)(inputs)
d = keras.layers.Conv2D(32, 8, 
        strides=4, activation='relu')(norm)
d = keras.layers.Conv2D(64, 4, strides=2, 
        activation="relu")(d)
d = keras.layers.Conv2D(64, 3, strides=1, 
        activation="relu")(d)
d = keras.layers.Flatten()(d)
d = keras.layers.Dense(512, activation='relu')(d)
output = keras.layers.Dense(action_size)(d)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

model.load_weights('./models/breakout_checkpoint')

decay_rate = 0.00001
gamma = 0.9

n_episodes = 50000
max_steps = 100
episode_render = True

decay_step = 1
loss = 0.0000
reward_list = []
for episode in range(n_episodes):
    state = env.reset()
    state = stack_frames(state, is_new=True)
    step = 0
    episode_rewards = []
    while step < max_steps:
        step += 1
        action = np.argmax(model.predict(state.reshape(1, *state.shape).astype(np.float32)))
        #action = 0
        #if step % 1000 == 0:
            #action = 1
        next_state, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        if episode_render:
            env.render()
        if done:
            total_reward = np.sum(episode_rewards)
            reward_list.append((episode, total_reward))
            break
        else:
            next_state = stack_frames(next_state)
            state = next_state

exit()
