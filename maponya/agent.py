import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
from autoencoder import MyAutoencoder
from env.env import FXEnv
from model import MyModel

class MyAgent:
    def __init__(self, model, sdae, state_size, action_size, env, nm='0'):
        self.nm = nm
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.sdae = sdae
        self.action_size = action_size
        self.state_size = state_size
        self.env = env
        self.last_obs = env.reset()

    def train(self, episode):
        batch_sz = 64
        observations = np.empty((batch_sz,) + self.state_size)
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        ep_reward = 0.0
        for step in range(batch_sz):
            self.sdae.noise_and_fit(self.last_obs[-1].reshape(1, -1))

            observations[step] = self.sdae.encode(self.last_obs.reshape(1, 60, 12))
            actions[step], values[step] = self.model.action_value(observations[step].reshape(1, *state_size))
            self.last_obs, rewards[step], dones[step], _ = self.env.step(actions[step])
            
            ep_reward += rewards[step]
            #if self.nm == '1':
                #self.env.render()

            if dones[step]:
                self.last_obs = self.env.reset()

        lastobs = self.sdae.encode(self.last_obs.reshape(1, 60, 12))
        _, next_value = self.model.action_value(lastobs)
        returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
        act_adv = np.concatenate((actions[:, None], advantages[:, None]), axis=-1)
        losses = self.model.train_on_batch(observations, [act_adv, returns])
        print('agent', self.nm, 'episode', episode)
        print('reward', ep_reward)
        print('loss', losses)

        if episode % 100 == 0 and episode != 0:
            self.model.save_weights('./weights/model')
            self.sdae.save_weights('./weights/sdae')
        
    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + returns[t + 1] * self.params['gamma']

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages


def value_loss(returns, values):
    return kls.mean_squared_error(returns, values) * params['value']

def logits_loss(act_adv, logits):
    ce = kls.CategoricalCrossentropy(from_logits=True)
    actions, advantages = tf.split(act_adv, 2, axis=-1)
    actions = tf.cast(actions, tf.int32)
    
    policy_loss = ce(actions, logits, sample_weight=advantages)
    entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)

    return policy_loss - entropy_loss * params['entropy']


params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }

pairs = [{'pair': 'gbpjpy', 'spread': 0.03}, {'pair': 'usdchf', 'spread': 0.0002}]
environments = []
agents = []

test_env = FXEnv('eurusd')
state_size = (60, 16)
action_size = 7

model = MyModel(action_size)
sdae = MyAutoencoder()

model.compile(optimizer = ko.Adam(lr=0.0001),
        loss=[logits_loss, value_loss])
sdae.compile(optimizer = ko.Adam(lr = 0.001), loss='mse')

for i in range(len(pairs)):
    environments.append(FXEnv(pairs[i]['pair'], spread=pairs[i]['spread']))
    agents.append(MyAgent(model, sdae, state_size, action_size, environments[i], nm=str(i)))

for i in range(200000):
    for agent in agents:
        agent.train(i)