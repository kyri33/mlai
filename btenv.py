import gym
from gym import spaces
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ochl as candlestick
from sklearn import preprocessing
import random
import pandas as pd

import gym


MAX_ACCOUNT_BALANCE = 2147483647
MAX_TRADING_SESSION = 4000  # ~2 months
MIN_TRADING_SESSSION = 1000 # 2 Weeks

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'

def date2num(date):
    return mdates.datestr2num(date)
    #converter = mdates.datestr2num('%Y-%m-%d')
    #return converter(date)

class StockTradingGraph:
    
    def __init__(self, df, title=None):
        self.df = df
        self.net_worths= np.zeros(len(df))

        fig = plt.figure()
        fig.suptitle(title)
        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0),
                rowspan=2, colspan=1)
        
        self.price_ax = plt.subplot2grid((6, 1), (2, 0),
                rowspan=8, colspan=1)
        
        self.volume_ax = self.price_ax.twinx()

        # padding
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90,
                top=0.90,wspace=0.2, hspace=0)
        plt.show(block=False) # Without blocking program !!!
    
    def render(self, current_step, net_worth, trades,
            window_size=40):
        
        self.net_worths[current_step] = net_worth
        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        dates = np.array([date2num(x) for x in self.df['Date'].values[step_range]])

        self._render_net_worth(current_step, net_worth, 
                step_range, dates)
        self._render_price(current_step, net_worth, dates, step_range)
        self._render_volume(current_step, net_worth, dates, step_range)
        self._render_trades(current_step, trades, step_range)

        self.price_ax.set_xticklabels(self.df['Date'].values[step_range],
                rotation=45, horizontalalignment='right')
            
        # Remove duplicate date labels in net worth
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
        
        # 'Necessary to view frames before  they are unrendered
        plt.pause(0.001)

    def _render_net_worth(self, current_step, net_worth, step_range,
            dates):
        self.net_worth_ax.clear()
        self.net_worth_ax.plot_date(dates, self.net_worths[step_range],
            '-', label='Net Worth')
        
        # Show Legend for net worth
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, 
                prop={'size':8})
        legend.get_frame().set_alpha(0.4)

        last_date = date2num(self.df['Date'].values[current_step])
        last_net_worth = self.net_worths[current_step]

        self.net_worth_ax.annotate('{0:.2f}'.format(net_worth),
            (last_date, last_net_worth), xytext=(last_date, last_net_worth),
            bbox = dict(boxstyle='round', fc='w', ec='k', lw=1),
            fontsize='small')
        
        self.net_worth_ax.set_ylim(min(self.net_worths[np.nonzero(self.net_worths)]) / 1.25,
                max(self.net_worths) * 1.25)
    
    def _render_price(self, current_step, net_worth, dates, step_range):
        self.price_ax.clear()

        candlesticks = zip(dates, self.df['Open'].values[step_range],
                        self.df['Close'].values[step_range],
                        self.df['High'].values[step_range],
                        self.df['Low'].values[step_range])
        candlestick(self.price_ax, candlesticks, width=1,
            colorup=UP_COLOR, colordown=DOWN_COLOR)
        
        last_date = date2num(self.df['Date'].values[current_step])
        last_close = self.df['Close'].values[current_step]
        last_high = self.df['High'].values[current_step]

        self.price_ax.annotate('{0:.2f}'.format(last_close),
            (last_date, last_close),
            xytext=(last_date, last_high),
            bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
            color='black', fontsize='small')
        
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                * VOLUME_CHART_HEIGHT, ylim[1])
        
    def _render_volume(self, current_step, net_worth, dates,
            step_range):
        
        self.volume_ax.clear()
        volume = np.array(self.df['Volume'].values[step_range])

        pos = self.df['Open'].values[step_range] - self.df['Close'].values[step_range] < 0
        neg = self.df['Open'].values[step_range] - self.df['Close'].values[step_range] > 0
        
        self.volume_ax.bar(dates[pos], volume[pos], color=UP_COLOR,
                alpha=0.4, width=1, align='center')
        self.volume_ax.bar(dates[neg], volume[neg], color=DOWN_COLOR,
                alpha=0.4, width=1, align='center')

        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, current_step, trades, step_range):
        for trade in trades:
            if trade['step'] in step_range:
                date = date2num(self.df['Date'].values[trade['step']])
                high = self.df['High'].values[trade['step']]
                low = self.df['Low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR
                
                total = '{0:.3f}'.format(trade['total'])
                self.price_ax.annotate(f'${total}', (date, high_low),
                    xytext=(date, high_low),
                    color = color,
                    fontsize=8,
                    arrowprops=(dict(color=color)))

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    scaler = preprocessing.MinMaxScaler()
    viewer = None

    def __init__(self, df, lookback_window_size=50, commission=0.00075,
                    initial_balance=10000, serial=False):
        super(StockEnv, self).__init__()
        self.df = df.dropna().reset_index()
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.serial = serial
        self.visualization = None
        self.current_step = 0
        # Action space Buy/Sell/Hold , 0.1/0.2/0.3/...
        self.action_space = spaces.MultiDiscrete([3, 10])

        self.observation_space = spaces.Box(low=0, high=1, shape=(10, lookback_window_size + 1),
                                                dtype=np.float16)

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.btc_held = 0

        self._reset_session()

        self.account_history = np.repeat([
            [self.net_worth],
            [0],
            [0],
            [0],
            [0]
        ], self.lookback_window_size + 1, axis=1)

        self.trades = []

        return self._next_observation()

    def _reset_session(self):
        self.current_step = 0

        if self.serial:
            self.steps_left = len(self.df) - self.lookback_window_size - 1
            self.frame_start = self.lookback_window_size
        else:
            self.steps_left = np.random.randint(MIN_TRADING_SESSSION, MAX_TRADING_SESSION)
            self.frame_start = np.random.randint(self.lookback_window_size, len(self.df) - self.steps_left)

        self.active_df = self.df[self.frame_start - self.lookback_window_size: self.frame_start + self.steps_left]

    def _next_observation(self):
        end = self.current_step + self.lookback_window_size + 1
        obs = np.array([
            self.active_df['Open'].values[self.current_step:end],
            self.active_df['High'].values[self.current_step:end],
            self.active_df['Low'].values[self.current_step:end],
            self.active_df['Volume'].values[self.current_step:end],
            self.active_df['Close'].values[self.current_step:end]
        ])
        scaled_history = self.scaler.fit_transform(self.account_history)
        obs = np.append(obs, scaled_history[:, -(self.lookback_window_size + 1):], axis=0)
        return obs

    def step(self, action):
        current_price = current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
        self._take_action(action, current_price)
        self.steps_left -= 1
        self.current_step += 1

        if self.steps_left == 0:
            self.balance += self.btc_held * current_price
            self.btc_held = 0
            self._reset_session()
        
        obs = self._next_observation()
        reward = self.net_worth
        done = self.net_worth <= 0
        
        return obs, reward, done, {}

    def _take_action(self, action, current_price):
        action_type = action[0]
        amount = action[1] / 10.0

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type < 1:
            btc_bought = self.balance / current_price * amount
            cost = btc_bought * current_price * (1 + self.commission)
            self.btc_held += btc_bought
            self.balance -= cost
        elif action_type < 2:
            btc_sold = self.btc_held * amount
            sales = btc_sold * current_price * (1 - self.commission)
            self.btc_held -= btc_sold
            self.balance += sales
        
        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({
                'step': self.current_step,
                'amount': btc_sold if btc_sold > 0 else btc_bought,
                'total': sales if btc_sold > 0 else cost,
                'type': 'sell' if btc_sold > 0 else 'buy'
            })
        
        self.net_worth = self.balance + self.btc_held * current_price
        self.account_history = np.append(self.account_history, [
            [self.net_worth],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def render(self, mode='live', title=None, **kwargs):
    
        if mode =='file':
            print("print to file TODO") #self._render_to_file(kwargs.get('filename', 'render.txt'))
        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(self.df, title)
            if self.current_step > self.lookback_window_size:
                self.visualization.render(self.current_step, self.net_worth, 
                        self.trades, window_size = self.lookback_window_size)

df = pd.read_csv('./datasets/AAPL.csv')
env = StockEnv(pd.read_csv('./datasets/AAPL.csv'))
env.reset()

for i in range(1000):
    action1 = np.random.randint(0, 2)
    action2 = np.random.randint(1, 11)
    env.step([action1, action2])
    env.render()