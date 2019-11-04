import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import warnings
import collections
from tensorflow import keras

env = retro.make('SpaceInvaders-Atari2600')
# state_size = env.observation_space
action_size = env.action_space.n
action_oh = np.array(np.identity(action_size, dtype=np.int).tolist())

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    # Normalise pixel values
    normalised_frame = cropped_frame/255.
    # Resize for 
    preprocessed_frame = transform.resize(normalised_frame, [110, 84])
    return preprocessed_frame

stack_size = 4
stacked_frames = collections.deque([np.zeros([110, 84], dtype=np.int) for i in range(stack_size)], 
                                        maxlen=stack_size)

def stack_frames(stacked_frames, state, is_new, stack_size):
    frame = preprocess_frame(state)
    if is_new:
        stacked_frames = collections.deque([np.zeros([110, 84], dtype=np.int) for i in range(stack_size)], 
                                        maxlen=stack_size)
        for _ in range(stack_size):
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
        return stacked_state, stacked_frames
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        return stacked_state, stacked_frames

state_size = [110, 84, 4]
learning_rate = 0.00025
total_episodes = 50 # total episodes per traning step ?
max_steps = 50000 # max steps per episode
batch_size = 64 # random batch sample size

# Exploration parameters
explore_start = 1.0 # probability to explore
explore_stop = 0.01
decay_rate = 0.00001

# Q Learning hyper parameters
gamma = 0.9 # Discounting rate

# Memory
pretrain_length = batch_size # size to intialize memory
memory_size = 1000000 # Total memory size

episode_render = True # If you want to 'render' the environment

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, 8, input_shape=tuple(state_size),
                strides=4, activation='elu'))
model.add(keras.layers.Conv2D(64, 4, strides=2, 
                            activation="elu"))
model.add(keras.layers.Conv2D(64,3,strides=2,activation="elu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation="elu"))
model.add(keras.layers.Dense(action_size, activation=None))
model.compile(loss="mse", optimizer = keras.optimizers.Adam(lr = learning_rate))

class Memory():
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indexes = np.random.choice(np.arange(buffer_size),
                    size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]
    
# Pretraining
memory = Memory(memory_size)
for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)

    choice = np.random.randint(0, action_size)
    action = action_oh[choice]
    next_state, reward, done, _ = env.step(action)
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)
    else:
        memory.add((state, action, reward, next_state, done))
        state = next_state

def predict_action(explore_start, explore_stop, 
                decay_rate, decay_step, state, action_oh):
    e = np.random.randn()
    # Better epsilon greedy ? research
    explore_prob = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_prob > e:
        choice = np.random.randint(0, action_size)
        action = action_oh[choice]
    else:
        Qs = model.predict(state.reshape((1, *state.shape)))
        choice = np.argmax(Qs)
        action = action_oh[choice]
    return action, explore_prob

decay_step = 0
reward_list = []
for episode in range(total_episodes):
    step = 0
    episode_rewards = []
    state = env.reset()

    state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)
    loss = 0.0000
    while step < max_steps:
        step += 1
        decay_step += 1
        action, explore_prob = predict_action(explore_start, 
                explore_stop, decay_rate, decay_step, 
                state, action_oh)
        next_state, reward, done, _ = env.step(action)
        if episode_render:
            env.render()
        episode_rewards.append(reward)
        if done:
            next_state = np.zeros((110, 84), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
            total_reward = np.sum(episode_rewards)
            print("Episode: {}".format(episode),
                "Total Reward: {}".format(total_reward),
                "Explore P: {:.4f}".format(explore_prob),
                "Training Loss {:.4f}".format(loss))
            reward_list.append((episode, total_reward))
            memory.add((state, action, reward, next_state, done))
            break
        else:
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
            memory.add((state, action, reward, next_state, done))
            state = next_state
        
        batch = memory.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        action_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])
        target_Qs_batch = []
        Qs_next_state = model.predict(next_states_mb)
        for i in range(len(batch)):
            terminal = dones_mb[i]
            if terminal:
                target = rewards_mb[i]
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                #print(target)
            target_Qs_batch.append(action_mb[i] * target)
        
        targets_mb = np.array([each for each in target_Qs_batch])
        history = model.fit(states_mb, targets_mb, verbose=0)
        loss = float(history.history['loss'][0])
    model.save_weights('./models/my_checkpoint')
        #exit()
