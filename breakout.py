import tensorflow as tf
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
import collections
from tensorflow import keras
import gym

env = gym.make("BreakoutDeterministic-v4")

action_size = env.action_space.n

def process_frame(frame):
    gray = np.mean(frame, axis=2).astype(np.uint8)
    gray = gray[::2, ::2]
    return gray

stack_size = 4
state_size = [105, 80, stack_size]
stacked_frames = collections.deque(maxlen=stack_size)
def stack_frames(state, stacked_frames, is_new=False):
    global stack_size
    state = process_frame(state)
    if is_new:
        for _ in range(stack_size):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)
    return stacked_frames, np.stack(stacked_frames, axis=2)
    

class Memory():
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen = max_size)
    
    def add(self, state):
        self.buffer.append(state)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indexes = np.random.choice(np.arange(buffer_size), 
            size=batch_size, replace=False)
        tempbuff = [self.buffer[i] for i in indexes]
        retbuffer = []
        for buff in tempbuff:
            retbuffer.append((buff[0].astype(np.float32), buff[1], buff[2], buff[3].astype(np.float32), buff[4]))
        return retbuffer

memory_size = 1000000
memory = Memory(memory_size)
batch_size = 64
for i in range(batch_size):
    if i == 0:
        state = env.reset()
        stacked_frames, state = stack_frames(state, stacked_frames, is_new = True)
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    if done:
        next_state = np.zeros(state.shape)
        stacked_frames, next_state = stack_frames(next_state, stacked_frames)
        memory.add((state, action, reward, next_state, done))
    else:
        stacked_frames, next_state = stack_frames(next_state, stacked_frames)
        memory.add((state, action, reward, next_state, done))
        state = next_state

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
model.summary()
decay_rate = 0.00001
gamma = 0.99

def pick_action(state, decay_step, e_start, e_stop):
    e = np.random.rand(1)[0]
    epsilon = e_stop + (e_start - e_stop) * np.exp(-decay_rate * decay_step)
    #epsilon = 1.0
    if epsilon > e:
        action = env.action_space.sample()
    else:
        action = np.argmax(np.array(model.predict_on_batch(state.reshape(1, *state.shape).astype(np.float32))))
    return action, epsilon

n_episodes = 3000
max_steps = 50000
episode_render = True
e_start = 1.0
e_stop = 0.01

decay_step = 1
loss = 0.0000
reward_list = []
for episode in range(n_episodes):
    state = env.reset()
    stacked_frames, state = stack_frames(state, stacked_frames, is_new=True)
    step = 0
    episode_rewards = []
    
    while step < max_steps:
        step += 1
        action, epsilon = pick_action(state, decay_step, e_start, e_stop)
        decay_step += 1
        next_state, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        if episode_render:
            env.render()
        if done:
            next_state = np.zeros((210, 160, 3)).astype(np.float)
            stacked_frames, next_state = stack_frames(next_state, stacked_frames)
            memory.add((state, action, reward, next_state, done))
            total_reward = np.sum(episode_rewards)
            reward_list.append((episode, total_reward))
            print("Episode: {}".format(episode),
                "Total Reward: {}".format(total_reward),
                "Explore P: {:.4f}".format(epsilon),
                "Training Loss {:.6f}".format(loss))
            break
        else:
            stacked_frames, next_state = stack_frames(next_state, stacked_frames)
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
        predict = np.array(model.predict_on_batch(next_states_mb[i].reshape(1, *next_states_mb[i].shape)))
        if terminal:
            target = -1
        else:
            target = rewards_mb[i] + gamma * np.max(predict)
        target_f = np.array(model.predict_on_batch(states_mb[i].reshape([1, *state_size])))
        target_f = target_f.reshape([4])
        target_f[actions_mb[i]] = target
        target_batch.append(target_f)
    target_batch = np.array(target_batch)
    history = model.fit(states_mb, target_batch, verbose=0)
    loss = float(history.history['loss'][0])
    #if episode % 50 == 0:
        #model.save_weights('./models/breakout_checkpoint')