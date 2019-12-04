import tensorflow as tf
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
from FXModel import FXModel
from env.FXEnv import FXEnv
import numpy as np

class A2CAgent:
    def __init__(self, model, state_size, action_size):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(optimizer = ko.Adam(lr=0.0001),
                loss=[self._logits_loss, self._value_loss])
        
        self.action_size = action_size
        self.state_size = state_size
    
    def train(self, env, episodes=10000):
        batch_sz = 64
        observations = np.empty((batch_sz,) + self.state_size)
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        self.env = env

        next_obs = env.reset()
        ep_rews = [0.0]
        ret_losses=[]
        losses = 0.0
        for episode in range(episodes):
            for step in range(batch_sz):
                observations[step] = next_obs
                actions[step], values[step] = self.model.action_value(next_obs.reshape(1, *self.state_size))
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])
                env.render()
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
                
            if episode > 0 and episode % 10 == 0:
                print("Episode:", episode, 'Reward', ep_rews[-1], 'losses', losses)

            print(ep_rews)
            _, next_value = self.model.action_value(next_obs.reshape(1, *self.state_size))
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            act_adv = np.concatenate((actions[:,None], advantages[:,None]), axis=-1)
            losses = self.model.train_on_batch(observations, [act_adv, returns])
            ret_losses.append(losses)

        return ep_rews, ret_losses

    def _value_loss(self, returns, values):
        return kls.mean_squared_error(returns, values) * self.params['value']
    
    def _logits_loss(self, act_adv, logits):
        ce = kls.CategoricalCrossentropy(from_logits=True)
        actions, advantages = tf.split(act_adv, 2, axis=-1)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce(actions, logits, sample_weight=advantages)
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - entropy_loss * self.params['entropy']
    
    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + returns[t + 1] * self.params['gamma'] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

model = FXModel(3)
env = FXEnv()
state_size = (60, 6)
agent = A2CAgent(model, state_size, 3)
rewards_history, losses = agent.train(env)