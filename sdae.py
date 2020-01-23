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
df['ICHI'] = ta.trend.ichimoku_a(df['High'], df['Low'], n1=9, n2=26, visual=False, fillna=True)

df = df.iloc[0:100000]
win_size = 1440

data_set = []

for i in tqdm(range(win_size + 1, len(df))):
    active_df = df.loc[i -win_size : i, ['Open', 'High', 'Low', 'Close', 
            'MACD', 'MA', 'EMA', 'ATR', 'ROC', 'ICHI']]
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(active_df)
    data_set.append(scaled_df[-1])

app = np.random.rand(len(data_set), 2)
data_set = np.append(np.array(data_set), app, axis = 1)
print(data_set.shape)

#for i in range(100):
#    print(data_set[i * 10: i * 10 + 11])

class MyAutoencoder(Model):
    def __init__(self):
        super().__init__(MyAutoencoder)
        input_size = 12
        hidden_size = 10
        code_size = 16
        
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


model = MyAutoencoder()
adm = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adm, loss='binary_crossentropy', metrics=['mae'])

X_train, X_test, y_train, y_test = train_test_split(data_set, data_set, test_size=0.2)

m = 0.0
std = 0.1
train_noise = np.random.normal(m, std, size=X_train.shape)
test_noise = np.random.normal(m, std, X_test.shape)

X_train_noisy = X_train + train_noise
X_test_noisy = X_test + test_noise

model.fit(X_train_noisy, X_train, batch_size=1, validation_data=(X_test, X_test), epochs=10)

predicted = model.predict(X_test_noisy)

print("err")
for i in range(len(X_test[0:10])):
    print(np.mean(np.abs(X_test[i] - predicted[i])))

print("noise")
for i in range(len(X_test[0:10])):
    print(np.mean(np.abs(X_test[i] - X_test_noisy[i])))

for i in range(10, 13):
    print("original")
    print(X_test[i])
    encoded = model.encode(X_test[i].reshape(1, 12))
    print("encoded")
    print(encoded)
    print("decoded")
    print(model.decode(encoded))