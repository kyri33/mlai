import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import skimage
from skimage import io
import collections
import sys
import argparse
import os

matplotlib.use('Agg')
parser = argparse.ArgumentParser()
parser.add_argument('-g', dest='gamma', type=float, required=True)
parser.add_argument('-v', dest='value', type=float, required=True)
parser.add_argument('-e', dest='entropy', type=float, required=True)
parser.add_argument('-l', dest='lr', type=float, required=True)
parser.add_argument('-b', dest='batch_size', type=int, required=True)
parser.add_argument('--name', dest='name', required=True)
parser.add_argument('--desc', dest='description', required=True)

args = parser.parse_args()

gamma = args.gamma
value = args.value
entropy = args.entropy
learning_rate = args.lr
batch_size = args.batch_size
render = False
name = args.name
description = args.description

dict = {
    "gamma": gamma,
    'value': value,
    'entropy': entropy,
    'learning_rate': learning_rate,
    'batch_size': batch_size
}

f = open("params.txt", "w+")

f.write("\n" + name + "\n\n")
f.write(description + "\n\n")
f.write("params:\n")

for key,val in dict.items():
    f.write("\t" + key + " : " + str(val) + "\n")

try:
    os.mkdir('models')
except:
    print("Models Directory exists")

def process_frame(frame):
    gray = np.mean(frame, axis=2)
    norm = gray / 255.
    crop = norm[50:-10, 5:-5]
    crop = crop[::2, ::2]
    return crop

stack_size = 4

stacked_frames = collections.deque(maxlen=stack_size)
def stack_frames(state, is_new=False):
    global stack_size
    global stacked_frames
    state = process_frame(state)
    if is_new:
        for _ in range(stack_size):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)
    return np.stack(stacked_frames, axis=2)

env = gym.make("BreakoutDeterministic-v4")
action_size = env.action_space.n

frame = env.reset()
processed_frame = stack_frames(frame, is_new=True)

state_size = processed_frame.shape

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(keras.Model):
    def __init__(self, action_size):
        super().__init__('mlp_policy')
        
        self.conv1 = kl.Conv2D(32, 8, strides=4, activation='relu')
        self.conv2 = kl.Conv2D(64, 4, strides=2, activation='relu')
        self.conv3 = kl.Conv2D(64, 3, strides=1, activation='relu')
        self.fl = kl.Flatten()
        #self.h1 = kl.Dense(512, activation='relu')
        #self.h2 = kl.Dense(512, activation='relu')
        self.h = kl.Dense(512, activation='relu')

        self.logits = kl.Dense(action_size)
        self.value = kl.Dense(1)
        self.dist = ProbabilityDistribution()
    
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        c = self.conv1(x)
        c = self.conv2(c)
        c = self.conv3(c)
        c = self.fl(c)
        #h_log = self.h1(c)
        #h_val = self.h2(c)
        h = self.h(c)
        return self.logits(h), self.value(h)
    
    def action_value(self, inputs):
        logits, value = self.predict_on_batch(inputs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

class A2CAgent:
    def __init__(self, model, state_size, action_size):
        self.params = {
            'gamma': gamma,
            'value': value,
            'entropy': entropy
        }
        self.model = model
        self.model.compile(
            optimizer = ko.Adam(lr=learning_rate),
            loss=[self._logits_loss, self._value_loss]
        )
        self.model.action_value(np.zeros((1, *state_size)))
        self.model.summary(print_fn=lambda x: f.write("\n" + x + '\n'))
        f.close()
        self.action_size = action_size
        self.state_size = state_size
    
    def train(self, env, episodes=10):
        batch_sz = batch_size
        observations = np.empty((batch_sz,) + self.state_size)
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        self.env = env
        state = env.reset()
        next_obs = stack_frames(state, is_new=True)
        ep_rews = [0.0]
        ret_losses = []
        losses = 0.0
        for episode in range(episodes):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs.reshape(1, *self.state_size))
                next_state, rewards[step], dones[step], _ = self.env.step(actions[step])
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    n_state = env.reset()
                    next_obs = stack_frames(n_state, is_new=True)
                else:
                    next_obs = stack_frames(next_state)
            if episode > 0 and episode % 2 == 0:
                print("Episode:", episode, "Reward:", ep_rews[-1], "Losses:", losses)

            _, next_value = self.model.action_value(next_obs.reshape(1, *self.state_size))

            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            act_adv = np.concatenate((actions[:,None], advantages[:,None]), axis=-1)
            losses = self.model.train_on_batch(observations, [act_adv, returns])
            ret_losses.append(losses)
            if episode % 2 == 0 and episode != 0:
                self.model.save_weights("./models/a2_breakout")
        return ep_rews, ret_losses
    
    def _value_loss(self, returns, values):
        return self.params['value'] * kls.mean_squared_error(returns, values)
    
    def _logits_loss(self, act_adv, logits):
        ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        actions, advantages = tf.split(act_adv, 2, axis=-1)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce(actions, logits, sample_weight=advantages)
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - entropy_loss * self.params['entropy']


    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                rewards[t] = -1
            else:
                returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

env = gym.make("BreakoutDeterministic-v4")
model = Model(env.action_space.n)
training = True

if training:
    #model.load_weights("./models/a2_breakout")
    agent = A2CAgent(model, state_size, action_size)
    rewards_history, losses = agent.train(env)
    #plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('rewards.png')
else:
    model.load_weights("./models/a2_breakout")
    for i in range(20):
        done = False
        obs = env.reset()
        obs = stack_frames(obs, is_new=True)
        while not done:
            action, _ = model.action_value(obs[None, :])
            obs, _, done, _ = env.step(action)
            obs = stack_frames(obs)
            env.render()