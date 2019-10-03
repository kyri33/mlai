import numpy as numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("datasets/jse/quotes.csv", sep=";")

def clean_commas(arr):
    return [float(val.replace(',', '.')) for val in arr]

data['Close'] = clean_commas(data['Close'])
data['Open'] = clean_commas(data['Open'])
data['High'] = clean_commas(data['High'])
data['Low'] = clean_commas(data['Low'])

data['Volume'] = [int(''.join(v.split())) for v in data['Volume']]

print(data.tail())

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_cols = ["Open", "High", "Low", "Close", "Volume"]
d_train, d_test = train_test_split(data, train_size = 0.8, test_size = 0.2, shuffle = False)

x_train = d_train.loc[:, train_cols].values
x_test = d_test.loc[:, train_cols].values

print(x_test.shape)

scaler = MinMaxScaler()

close_col = 3

smoothing_window_size = 250

for di in range(0, x_train.shape[0], smoothing_window_size):
    scaler.fit(x_train[di:di+smoothing_window_size,:])
    x_train[di:di+smoothing_window_size,:] = scaler.transform(x_train[di:di+smoothing_window_size,:])

# scaler.fit(x_train[di + smoothing_window_size:, :])
# x_train[di + smoothing_window_size:, :] = scaler.transform(x_train[di+smoothing_window_size:, :])
x_test = scaler.transform(x_test)

# HYPER PARAMETERS

D = 5 # Dimensionality of the data
num_unrollings = 30 # How far to look into the future
batch_size = 50 # Number of samples in a batch
num_nodes = [200] # Num of nodes in each hidden layer of the network
n_layers = len(num_nodes) # Number of hidden layers
dropout = 0.2 # dropout amount

tf.reset_default_graph()

train_inputs, train_outputs = [], []
# Input and output placeholders unrolled over time for each step ?

for ui in range()