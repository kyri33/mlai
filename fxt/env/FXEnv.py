import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
import random
import fxdata
from sklearn import preprocessing
import pandas as pd
import random
from FXGraph import FXGraph

class FXEnv(gym.Env):

    # TODO Decide what to do with start date

    def __init__(self,  commission=0.00075,
            initial_balance=100000):
        super(FXEnv, self).__init__()

        self.group_by = fxdata.DAY
        self.initial_balance = initial_balance
        self.commission = commission
        self.visualization = None

        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space = spaces.Box(low=0, high=1, shape=[10], dtype=np.float16)

        self.data = fxdata.load_data(fxdata.DAY)
        self.total_sets = len(self.data)
        self.current_set = 0

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.holding = 0
        self.current_set += 1
        if self.current_set >= self.total_sets:
            self.current_set = 1
        
        self.current_min = 0
        self.mins_left = len(self.data[0])

        self.account_history = np.array([
                [self.net_worth],[0],[0],[0],[0]])
        
        self.trades = []

        self.active_df = self.data[self.current_set][['Date', 'Open', 'High', 'Low', 'Close']]

        return self._next_observation()
    
    def _next_observation(self):

        # TODO Test scaling

        window_size = self.group_by
        scaled_features = ['Open', 'High', 'Low', 'Close']
        prev_df = self.data[self.current_set - 1]
        scale_df = pd.concat([
            prev_df.iloc[-(window_size - self.current_min):],
            self.active_df.iloc[:self.current_min]
        ])
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(scale_df[scaled_features])
        obs = scaler.transform(self.active_df[scaled_features].iloc[self.current_min].values.reshape(1, -1))
        
        return obs

    def step(self, action):
        current_price = random.uniform(
            self.active_df.loc[self.current_min, 'Open'],
            self.active_df.loc[self.current_min, 'Close']
        )
        self._take_action(action, current_price)
        self.mins_left -= 1
        self.current_min += 1

        if self.mins_left == 0:
            # TODO Remember profits ?
            obs = self.reset()
    
    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10.0

        bought = 0
        sold = 0
        cost = 0
        sales = 0

        if action_type < 1:
            bought = self.balance / current_price * amount
            cost = bought * current_price * (1 + self.commission)
            self.holding += bought
            self.balance -= cost
        elif action_type < 2:
            sold = self.holding * amount
            sales = sold * current_price * (1 - self.commission)
            self.holding -= sold
            self.balance += sales
        
        if sold > 0 or bought > 0:
            self.trades.append({
                'step': self.current_min,
                'amount': sold if sold > 0 else bought,
                'total': sales if sold > 0 else cost,
                'type': 'sell' if sold > 0 else 'buy'
            })
        
        self.net_worth = self.balance + self.holding * current_price
        self.account_history = np.array([
            [self.net_worth],
            [bought],
            [cost],
            [sold],
            [sales]
        ], axis=1)

    def render(self, mode='live', title=None, **kwargs):
        if mode == 'file':
            # TODO print to file to watch later
            print("Print File TODO")
        elif mode == 'live':
            if self.visualization == None:
                self.visualization = FXGraph(self.active_df, title)
                



            

env = FXEnv()
env.reset()
env.render()