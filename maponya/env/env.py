import gym
from gym import spaces
from env import fdata
from sklearn import preprocessing
import numpy as np
import random
from env import mypreprocessing
import pandas as pd
from env.graph import ENVGraph

#import keyboard

#from autoencoder import MyAutoencoder
#from random import randrange
#import tensorflow.keras as keras

class FXEnv(gym.Env):

    def __init__(self, pair, spread=0.0,
            initial_balance=10000, look_back=60):
        
        super(FXEnv, self).__init__()

        pd.options.mode.chained_assignment = None
        self.group_by = fdata.DAY
        self.look_back = look_back
        self.initial_balance = initial_balance
        self.spread = spread

        self.year = 2011
        self.maxyear = 2018
        
        self.action_space = spaces.Discrete(7)
        self.maxaction = 6
        self.observation_space = spaces.Box(low=0, high=1, shape=[self.look_back, 12])
        self.pair = pair
        self.data = fdata.load_data(self.pair, self.year)
        self.current_day = 0
        self.position_amount = initial_balance * 0.2
        self.initial_portfolio = self.position_amount * (self.action_space.n // 2)

        self.pos_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.pos_scaler.fit(np.array([[-3], [3]]))
        self.totalDays = len(self.data) // fdata.DAY

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_day += 1
        if self.current_day >= self.totalDays:
            self.current_day = 1
            self.year += 1
            if self.year > self.maxyear:
                self.year = 2011
            self.data = fdata.load_data(self.pair, self.year)

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
        self.trades = []
        self.visualization = None

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
        
        obs_df = self.data.iloc[self.cur_step - self.look_back + 1 : self.cur_step + 1].copy()
        obs_df[minmaxindex] = scaler.transform(obs_df[minmaxindex])
        obs_df[customindex] = myscaler.transform(obs_df[customindex].values)
        
        mn = np.mean(self.returns)
        std = np.std(self.returns)
        if std == 0:
            sharpe = 0.
        else:
            sharpe = mn / std

        pos = self.pos_scaler.transform(np.array([[self.prev_action]]))[0][0]

        self.private = np.append(self.private, np.array([[pos, sharpe]]), axis=0)
        sharpscaler = mypreprocessing.CustomMinMaxScaler()
        self.private[:,1] = sharpscaler.fit_transform(self.private[:,1].reshape(-1, 1)).reshape(-1)

        obs = np.append(obs_df, self.private[-self.look_back:], axis=1)
        
        return obs

    def step(self, action):
        action = action - 3
        current_price = random.uniform(
            self.data.loc[self.cur_step, 'Open'],
            self.data.loc[self.cur_step, 'Close']
        )
        current_position = self._take_action(action, current_price)
        self.prev_action = action
        self.net_worth = self.balance + current_position * current_price
        self.mins_left -= 1
        self.current_min += 1
        self.cur_step += 1
        if self.prev_price == 0:
            self.prev_price = current_price
        
        # TODO ADD COMMISSION AND SLIPPAGE
        profit = (current_price - self.prev_price) * self.prev_position
        reward = profit - abs(current_position - self.prev_position) * self.spread
        self.prev_position = current_position
        self.prev_price = current_price
        #profit = (self.net_worth + (self.initial_portfolio - self.initial_balance) - self.initial_portfolio) / self.initial_portfolio
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
        short_sales = 0
        cover_cost = 0

        if self.net_worth <= (self.maxaction - 3) * self.position_amount:
            self.balance = self.initial_balance

        if action == self.prev_action:
            return self.prev_position

        if action > self.prev_action:
            if self.prev_action < 0:
                covered, cover_cost = self._cover(current_price, action - self.prev_action)
                action += self.prev_action
            if action >= 0:
                bought, cost = self._long(current_price, action - self.prev_action)
        elif action < self.prev_action:
            if self.prev_action > 0:
                sold, sales = self._sell(current_price, action - self.prev_action)
                action += self.prev_action
            if action <= 0:
                shorted, short_sales = self._short(current_price, action - self.prev_action)
        
        sales += short_sales
        cost += cover_cost
        current_position = self.prev_position - shorted + covered - sold + bought
        self.balance = self.balance + sales - cost

        if sales > 0 or cost > 0:
            self.trades.append({
                'step': self.cur_step,
                'amount': sold if sold > 0 else bought,
                'total': sales if sales > 0 else cost,
                'type': 'sell' if sales > 0 else 'buy'
            })

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

    def render(self, mode='live', title=None, **kwargs):
        
        if mode == 'file':
            print("File TODO")
        elif mode == 'live':
            if self.visualization == None:
                self.visualization = ENVGraph(self.data, title)
            else:
                self.visualization.render(self.cur_step, self.net_worth, 
                        self.trades, self.balance, self.prev_position)


'''
env = FXEnv('gbpjpy')
env2 = FXEnv('usdchf')
test_env = FXEnv('')
env.reset()
env2.reset()
test_env.reset()

step = 3
print('begun')
encoder = MyAutoencoder()
adm = keras.optimizers.Adam(learning_rate=0.001)
encoder.compile(optimizer=adm, loss='mse', metrics=['mae'])
for k in range(100000):

    key = keyboard.read_key()
    if key == '0':
        step = 0
    elif key == '1':
        step = 1
    elif key == '2':
        step = 2
    elif key == '3':
        step = 3
    elif key == '4':
        step = 4
    elif key == '5':
        step = 5
    elif key == '6':
        step = 6
    step = randrange(6)
    step2 = randrange(6)
    test_step = randrange(6)
    state, reward, _, _ = env.step(step)
    state2, r, _, _ = env2.step(step2)
    test_state, r, _, _ = test_env.step(test_step)

    m = 0.0
    std = 0.1
    noise = np.random.normal(m, std, size=state.shape)
    state_noisy = state + noise

    noise2 = np.random.normal(m, std, size=state2.shape)
    state_noisy2 = state2 + noise2

    jstate = np.append(state_noisy[-1].reshape(1, -1), state_noisy2[-1].reshape(1, -1), axis=0)
    nstate = np.append(state[-1].reshape(1, -1), state2[-1].reshape(1, -1), axis=0)
    print(k)
    encoder.fit(jstate, nstate, batch_size=1, epochs=1, validation_data=(test_state, test_state))
'''