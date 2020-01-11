import gym
from gym import spaces
import fdata
from sklearn import preprocessing
import numpy as np

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
        self.position_amount = initial_balance * 0.2

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.holding = 0
        self.current_day += 1
        self.current_min = 0
        self.mins_left = fdata.DAY
        self.trades = []
        self.anchor = fdata.DAY * self.current_day
        self.step = self.anchor + self.current_min
        self.prev_price = 0
        self.prev_action = 0
        self.prev_position = 0

        return self._next_observation()

    def _next_observation(self):
        window_size = self.group_by
        scale_df = self.data[self.step - window_size + 1 : self.step + 1]
        
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(scale_df)

        obs_df = self.data.iloc[self.step - self.look_back + 1 : self.step + 1]
        obs_df = scaler.transform(obs_df)

    def step(self, action):
        current_price = random.uniform(
            self.data.loc[self.step, 'Open'],
            self.data.loc[self.step, 'Close']
        )
        current_position = self._take_action(action, current_price)
        self.mins_left -= 1
        self.current_min += 1
        self.step += 1
        if self.prev_price == 0:
            self.prev_price = current_price
        
        # TODO ADD COMMISSION AND SLIPPAGE
        reward = (current_price - self.prev_price) * self.prev_position - abs(current_position - self.prev_position)
        self.prev_position = current_position

        if self.mins_left == 0:
            done = True
            obs = np.zeros((60, 12))
        else:
            obs = self._next_observation()
            done = False
        
        return obs, reward, done, {}

    def _take_action(self, action, current_price):
        action = action - 3
        bought = 0
        sold = 0
        shorted = 0
        covered = 0
        cost = 0
        sales = 0

        if action == self.prev_action:
            return self.prev_position

        if action > self.prev_action:
            if self.prev_action < 0:
                covered, sales += self._cover(current_price, action - self.prev_action)
            if action > 0:

            
    
    def _cover(current_price, action_diff):
        total = 0
        units = 0
        if -action_diff < self.prev_action:
            units = self.prev_position
            total = self.prev_position * current_price
        else:
            total = self.position_amount * action_diff
            units = total / current_price
        
        return abs(units), total

env = FXEnv()
env.reset()