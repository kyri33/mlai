import gym
from gym import spaces
import fdata
from sklearn import preprocessing

class FXEnv(gym.Env):

    def __init__(self, commission=0.0,
            initial_balance=10000, look_back=60):
        
        super(FXEnv, self).__init__()

        self.group_by = fdata.DAY
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.commission = commission
        
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1, shape=[self.look_back, 12])
        self.data = fdata.load_data()
        self.current_day = 0

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_position = 0
        self.holding = 0
        self.current_day += 1
        self.current_min = 0
        self.mins_left = fdata.DAY
        self.trades = []
        self.anchor = fdata.DAY * self.current_day
        self.step = self.anchor + self.current_min

        return self._next_observation()

    def _next_observation(self):
        window_size = self.group_by
        scale_df = self.data[self.step - window_size + 1 : self.step + 1]
        
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(scale_df)

        obs_df = self.data[self.step - self.look_back + 1 : self.step + 1]
        
    

env = FXEnv()
env.reset()