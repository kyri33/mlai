import gym
from gym import spaces
import numpy as np
from sklearn import preprocessing
import random
from env import fxdata
from sklearn import preprocessing
import pandas as pd
import random
from env.FXGraph import FXGraph

class FXEnv(gym.Env):

    # TODO Decide what to do with start date

    def __init__(self,  commission=0.0,
            initial_balance=10000, look_back = 60):
        super(FXEnv, self).__init__()

        self.group_by = fxdata.DAY
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.commission = commission
        self.visualization = None

        #self.action_space = spaces.MultiDiscrete([3, 3])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=[self.look_back, 4], dtype=np.float16)

        self.data = fxdata.load_data(fxdata.DAY)
        self.total_sets = len(self.data)
        self.current_set = 0

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_networth = self.net_worth
        self.holding = 0
        self.current_set += 1
        if self.current_set >= self.total_sets:
            self.current_set = 1
        
        self.current_min = 0
        self.mins_left = len(self.data[0])

        self.account_history = np.repeat([
            [self.balance, self.holding]
        ], self.look_back, axis=0)

        self.trades = []

        self.active_df = self.data[self.current_set][['Date', 'Open', 'High', 'Low', 'Close']]
        self.active_df.reset_index(inplace=True)

        return self._next_observation()
    
    def _next_observation(self):

        # TODO Test scaling

        window_size = self.group_by
        scaled_features = ['Open', 'High', 'Low', 'Close']
        prev_df = self.data[self.current_set - 1]

        if self.current_min < window_size - 1:
            scale_df = pd.concat([
                prev_df.iloc[-(window_size - self.current_min - 1):],
                self.active_df.iloc[:self.current_min + 1]
            ], sort=False)
        else:
            scale_df = self.active_df.iloc[:self.current_min + 1]

        scaler = preprocessing.MinMaxScaler()
        scaler.fit(scale_df[scaled_features])

        account_scaler = preprocessing.MinMaxScaler()
        account_scaled = account_scaler.fit_transform(self.account_history)

        if self.current_min < self.look_back - 1:
            n = self.look_back - self.current_min - 1
            obs_df = pd.concat([
                prev_df.iloc[-n:],
                self.active_df.iloc[:self.current_min + 1]
            ], sort=False)
        else:
            obs_df = self.active_df.iloc[self.current_min - self.look_back + 1 : self.current_min + 1]
        
        obs = scaler.transform(obs_df[scaled_features].values)

        obs = np.append(obs, account_scaled[-self.look_back:], axis=1)
        
        return obs

    def step(self, action):
        current_price = random.uniform(
            self.active_df.loc[self.current_min, 'Open'],
            self.active_df.loc[self.current_min, 'Close']
        )
        self._take_action(action, current_price)
        self.mins_left -= 1
        self.current_min += 1

        reward = self.net_worth - self.prev_networth
        self.prev_networth = self.net_worth

        if self.mins_left == 0:
            # TODO Remember profits ?
            done = True
            obs = np.zeros((60, 6))
        else:
            obs = self._next_observation()
            done = self.net_worth < self.initial_balance / 3

        return obs, reward, done, {}
    
    def _take_action(self, action, current_price):
        action_type = action
        #amount = action[1] / 10.0
        amount = 0.3

        bought = 0
        sold = 0
        cost = 0
        sales = 0

        if action_type < 1:
            #bought = self.balance / current_price * amount
            if self.balance > 1000:
                bought = 1000 / current_price
                cost = bought * current_price * (1 + self.commission)
                self.holding += bought
                self.balance -= cost
        elif action_type < 2:
            #sold = self.holding * amount
            sold = self.holding
            if self.holding * current_price > 2000:
                sold = 1000 / current_price

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
        self.account_history = np.append(self.account_history, np.array([
            [self.balance, self.holding]
        ]), axis=0)

    def render(self, mode='live', title=None, **kwargs):
        if mode == 'file':
            # TODO print to file to watch later
            print("Print File TODO")
        elif mode == 'live':
            if self.visualization == None:
                self.visualization = FXGraph(self.active_df, title)
            else:
                self.visualization.render(self.current_min, self.net_worth,
                        self.trades, self.balance, self.holding)
                
'''          
env = FXEnv()
env.reset()
env.render()
for i in range(500):
    action1 = 2
    if i < 40 and i > 10:
        action1 = 0
    elif i > 150:
        action1 = 1
    #action1 = np.random.randint(0, 2)
    action2 = np.random.randint(1, 4)
    obs, reward, done, _ = env.step([action1, action2])
    print(reward)
    env.render()
'''