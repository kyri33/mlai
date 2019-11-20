import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras as keras
import logging
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')

        self.value = kl.Dense(1, name='value')
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()
    
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)

        return self.logits(hidden_logs), self.value(hidden_vals)
    
    def action_value(self, inputs):
        logits, value = self.predict_on_batch(inputs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

arr = np.random.rand(1, 8)
model = Model(4)
a, v = model.action_value(arr)

checkpoint = "cartpole_checkpoint2"

class A2CAgent:
    def __init__(self, model):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer = ko.RMSprop(lr=0.0007),
            loss=[self._logits_loss, self._value_loss]
        )
        self.env = gym.make("CartPole-v0")
    
    def train(self, batch_sz=32, updates=100):
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty(((batch_sz,) + self.env.observation_space.shape))

        ep_rews=[0.0]
        ep_loss = []
        next_obs = self.env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None,:])

                next_obs, rewards[step], dones[step], _ = self.env.step(actions[step])
                self.env.render()
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = self.env.reset()

            print("Episode:", update, "Reward:", ep_rews[-1])            
            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            act_adv = np.concatenate((actions[:, None], advantages[:, None]), axis=-1)
            losses = self.model.train_on_batch(observations, [act_adv, returns])
            ep_loss.append(losses)
            if update % 25 == 0:
                self.model.save_weights('./models/' + checkpoint)
        return ep_rews, ep_loss

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), [next_value], axis=-1)

        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1-dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, values):
        return self.params['value'] * tf.keras.losses.mean_squared_error(returns, values)

    def _logits_loss(self, act_adv, logits):
        actions, advantages = tf.split(act_adv, 2, axis=-1)
        
        cross_entropy = kls.SparseCategoricalCrossentropy(from_logits=True)

        actions = tf.cast(actions, tf.int32)
        policy_loss = cross_entropy(actions, logits, sample_weight=advantages)
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)

        return policy_loss - entropy_loss * self.params['entropy']

logging.getLogger().setLevel(logging.INFO)

env = gym.make("CartPole-v0")
model = Model(env.action_space.n)

training = True

if training:
    agent = A2CAgent(model)

    procs = []
    agents = []
    '''
    for i in range(multiprocessing.cpu_count()):
        print('starting')
        agent = A2CAgent(model)
        agents.append(agent)
        process = Process(target=agent.train)
        procs.append(process)
        process.start()
    print('done')
    for proc in procs:
        proc.join()
    '''

    rewards_history, losses=agent.train()

    plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    losses = np.array(losses)
    plt.plot(range(0, len(losses[5:])), losses[5:, 1:])
    plt.show()

model.load_weights("./models/" + checkpoint)
for i in range(20):
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.action_value(obs[None, :])
        obs, _, done, _ = env.step(action)
        env.render()