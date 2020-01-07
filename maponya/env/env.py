import gym
from gym import spaces
from env import data

class FXEnv(gym.Env):

    def __init__(self, commission=0.0,
            initial_balance=10000, look_back=60):
        
        super(FXEnv, self).__init__()

        self.group_by = data.DAY
        self.look_back = look_back
        self.initial_balance = intial_balance
        self.commission = commission
        
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1, shape=[self.look_back, ])