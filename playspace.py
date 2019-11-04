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
model.load_weights('./models/my_checkpoint')
state = env.reset()
state, stacked_frames = stack_frames(stacked_frames, state, True, 4)
for step in range(100000):
    model_c = model.predict(state.reshape(1, *state.shape))
    print(model_c)
    choice = np.argmax(model_c)
    #if step < 250:
    #    action = action_oh[7]
    #else:
    #    choice = step % 8
    #    action = action_oh[1]
    action = action_oh[choice]
    #print("Action:", action)
    next_state, reward, done, _ = env.step(action)
    env.render()
    state, stacked_frames = stack_frames(stacked_frames, next_state, False, 4)
    if (done):
        break