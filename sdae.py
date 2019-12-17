import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import tensorflow.keras as keras
import tensorflow.keras.datasets.mnist as mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import ta

df = pd.read_csv('./fxt/env/datasets/fx_2018.csv', sep=';', names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

ind_macd = ta.trend.macd(df['Close'], n_slow=52, n_fast=25, fillna=True)
ind_ma = ta.volatility.bollinger_mavg(df['Close'], n=20, fillna=True)
ind_ema = ta.trend.ema_indicator(df['Close'], n=12, fillna=True)
ind_atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], n=14, fillna=True)
ind_roc = ta.momentum.roc(df['Close'], n=12, fillna=True)

df['MACD'] = ind_macd
df['MA'] = ind_ma
df['EMA'] = ind_ema
df['ATR'] = ind_atr
df['ROC'] = ind_roc

print(df)
win_size = 1440

data_set = []

for i in tqdm(range(win_size + 1, len(df))):
    active_df = df.loc[i -win_size : i, ['Open', 'High', 'Low', 'Close', 
            'MACD', 'MA', 'EMA', 'ATR', 'ROC']]
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(active_df)
    data_set.append(scaled_df[-1])

data_set = np.array(data_set)
print(data_set.shape[0] - len(df))

class MyAutoencoder(Model):
    def __init__(self):
        super().__init__(MyAutoencoder)
        input_size = x_train.shape[1]
        hidden_size = 128
        code_size = 32
        
        self.hidden1 = Dense(hidden_size, activation='relu')
        self.code = Dense(code_size, activation='relu')
        self.hidden2 = Dense(hidden_size, activation='relu')
        self.os = Dense(input_size, activation='sigmoid')
    
    def encode(self, inputs):
        x = tf.convert_to_tensor(inputs)
        h = self.hidden1(x)
        return self.code(h)

    def decode(self, codes):
        h = self.hidden2(codes)
        return self.os(h)
    
    def call(self, inputs):
        code = self.encode(inputs)
        return self.decode(code)