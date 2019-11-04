import tensorflow as tf
import numpy as np
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
    epsilon = np.exp(-decay_rate * decay_step)
    #epsilon = 0.0
    if epsilon > e:
        action = env.action_space.sample()
    else:
        action = np.argmax(model.predict(state.reshape(1, *state.shape)))
    return action, epsilon

n_episodes = 50000
max_steps = 50000
episode_render = False

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
            total_reward = np.sum(episode_rewards)
            reward_list.append((episode, total_reward))
            print("Episode: {}".format(episode),
                "Total Reward: {}".format(total_reward),
                "Explore P: {:.4f}".format(epsilon),
                "Training Loss {:.4f}".format(loss))
            break
        else:
            next_state = stack_frames(next_state)
            memory.add((state, action, reward, next_state, done))
            state = next_state
    
    batch = memory.sample(batch_size)
    states_mb = np.array([each[0] for each in batch])
    actions_mb = np.array([each[1] for each in batch])
    rewards_mb = np.array([each[2] for each in batch])
    next_states_mb = np.array([each[3] for each in batch])
    dones_mb = np.array([each[4] for each in batch])

    target_batch = []
    for i in range(batch_size):
        terminal = dones_mb[i]
        predict = model.predict(next_states_mb[i].reshape(1, *next_states_mb[i].shape))
        if terminal:
            target = rewards_mb[i]
        else:
            target = rewards_mb[i] + gamma * np.max(predict)
        target_f = model.predict(states_mb[i].reshape([1, *state_size]))
        target_f = target_f.reshape([4])
        target_f[actions_mb[i]] = target
        target_batch.append(target_f)
    target_batch = np.array(target_batch)
    history = model.fit(states_mb, target_batch, verbose=0)
    loss = float(history.history['loss'][0])
    if episode % 1000 == 0:
        model.save_weights('./models/breakout_checkpoint')