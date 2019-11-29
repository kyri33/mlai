import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
import random

class FXEnv(gym.Env):

    def __init__(self,  commission=0.00075,
            initial_balance=10000):
        super(FXEnv, self).__init__()

        self.initial_balance = initial_balance
        self.commission = commission
        self.action_space = spaces.MultiDiscrete([3, 10])
        self.observation_space = spaces.Box(low=0, high=1, shape=(10), dtype=np.float16)