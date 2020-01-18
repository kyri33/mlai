import gym
from gym import spaces
import fdata
from sklearn import preprocessing
import numpy as np
import random
import mypreprocessing
import pandas as pd

class FXEnv(gym.Env):

    def __init__(self, spread=0.0,
            initial_balance=10000, look_back=60):
        
        super(FXEnv, self).__init__()

        pd.options.mode.chained_assignment = None
        self.group_by = fdata.DAY
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.spread = spread
        
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1, shape=[self.look_back, 12])
        self.data = fdata.load_data()
        self.current_day = 0
        self.position_amount = initial_balance * 0.2
        self.initial_portfolio = self.position_amount * (self.action_space.n // 2)

        self.pos_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.pos_scaler.fit(np.array([[-3], [3]]))

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.holding = 0
        self.current_day += 1
        self.current_min = 0
        self.mins_left = fdata.DAY
        self.trades = []
        self.anchor = fdata.DAY * self.current_day
        self.cur_step = self.anchor + self.current_min
        self.prev_price = 0
        self.prev_action = 0
        self.prev_position = 0
        self.returns = np.zeros((self.look_back))
        self.private = np.zeros((self.look_back, 2))

        return self._next_observation()

    def _next_observation(self):
        window_size = self.group_by
        scale_df = self.data.iloc[self.cur_step - window_size + 1 : self.cur_step + 1]
        minmaxindex = ['Open', 'Close', 'High', 'Low', 'MA', 'EMA', 'ATR', 'ICHI']
        customindex = ['MACD', 'ROC']

        scaler = preprocessing.MinMaxScaler()
        scaler.fit(scale_df[minmaxindex])
        myscaler = mypreprocessing.CustomMinMaxScaler()
        myscaler.fit(scale_df[customindex].values)

        obs_df = self.data.iloc[self.cur_step - self.look_back + 1 : self.cur_step + 1]
        obs_df[minmaxindex] = scaler.transform(obs_df[minmaxindex])
        obs_df[customindex] = myscaler.transform(obs_df[customindex].values)

        mn = np.mean(self.returns[-self.look_back:])
        std = np.std(self.returns[-self.look_back:])
        if std == 0:
            sharpe = 0.
        else:
            sharpe = mn / std
        pos = self.pos_scaler.transform(np.array([[self.prev_action]]))[0][0]

        self.private = np.append(self.private[-self.look_back + 1:], np.array([[pos, sharpe]]), axis=0)
        sharpscaler = mypreprocessing.CustomMinMaxScaler()
        self.private[:,1] = sharpscaler.fit_transform(self.private[:,1].reshape(-1, 1)).reshape(-1)

        obs = np.append(obs_df, self.private, axis=1)
        print(obs)
        return obs

    def step(self, action):
        action = action - 3
        current_price = random.uniform(
            self.data.loc[self.cur_step, 'Open'],
            self.data.loc[self.cur_step, 'Close']
        )
        current_position = self._take_action(action, current_price)
        self.prev_action = action
        self.mins_left -= 1
        self.current_min += 1
        self.cur_step += 1
        if self.prev_price == 0:
            self.prev_price = current_price
        
        # TODO ADD COMMISSION AND SLIPPAGE
        reward = (current_price - self.prev_price) * self.prev_position - abs(current_position - self.prev_position) * self.spread
        self.prev_position = current_position
        profit = (self.net_worth + (self.initial_portfolio - self.initial_balance) - self.initial_portfolio) / self.initial_portfolio
        self.returns = np.append(self.returns, profit)

        if self.mins_left == 0:
            done = True
            obs = np.zeros((self.look_back, 12))
        else:
            obs = self._next_observation()
            done = False
        
        return obs, reward, done, {}

    def _take_action(self, action, current_price):
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
                covered, cost = self._cover(current_price, action - self.prev_action)
                action += self.prev_action
            if action > 0:
                bought, cost = self._long(current_price, action - self.prev_action)
        elif action < self.prev_action:
            if self.prev_action > 0:
                sold, sales = self._sell(current_price, action - self.prev_action)
                action += self.prev_action
            if action < 0:
                shorted, sales = self._short(current_price, action - self.prev_action)
        
        current_position = self.prev_position - shorted + covered - sold + bought
        self.balance = self.balance + sales - cost
        self.net_worth = self.balance + self.prev_position * current_price
        return current_position

    def _sell(self, current_price, action_diff):
        total = 0
        units = 0
        if -action_diff >= self.prev_action:
            units = self.prev_position
            total = self.prev_position * current_price
        else:
            total = action_diff * self.position_amount
            units = total / current_price
        return abs(units), abs(total)

    def _cover(self, current_price, action_diff):
        total = 0
        units = 0
        if -action_diff <= self.prev_action:
            units = self.prev_position
            total = self.prev_position * current_price
        else:
            total = self.position_amount * action_diff
            units = total / current_price
        
        return abs(units), abs(total)
    
    def _long(self, current_price, action_diff):
        total = action_diff * self.position_amount
        units = total / current_price
        return abs(units), abs(total)
    
    def _short(self, current_price, action_diff):
        total = action_diff * self.position_amount
        units = total / current_price
        return abs(units), abs(total)



env = FXEnv()
env.reset()