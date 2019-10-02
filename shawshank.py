import pandas as pd
import matplotlib.pyplot as plt
import string
import numpy as np
from tqdm import tqdm
import tensorflow as tf

data = pd.read_csv("datasets/jse/quotes.csv", sep=";")

def clean_commas(arr):
    return [float(val.replace(',', '.')) for val in arr]

data['Close'] = clean_commas(data['Close'])
data['Open'] = clean_commas(data['Open'])
data['High'] = clean_commas(data['High'])
data['Low'] = clean_commas(data['Low'])

print(''.join(data['Volume'][0].split()))
data['Volume'] = [int(''.join(v.split())) for v in data['Volume']]

print(data.tail())

#plt.plot(data['Close'], 'r')
#plt.plot(data['Volume'], 'b')
#plt.show()


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_cols = ["Open", "High", "Low", "Close", "Volume"]
d_train, d_test = train_test_split(data, train_size = 0.8, test_size = 0.2, shuffle = False)

x = d_train.loc[:, train_cols].values
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x)
print(x_train.shape)
x_test = scaler.transform(d_test.loc[:, train_cols].values)

TIME_STEPS = 10
BATCH_SIZE = 30

def build_timeseries(mat, y_col_index): # Puts values into 3D array
    dim_0 = mat.shape[0] - TIME_STEPS - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0, TIME_STEPS, 1))

    for i in tqdm(range(dim_0)):
        x[i] = mat[i : TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i : TIME_STEPS + i + TIME_STEPS, y_col_index].reshape(-1, 1)
    print("length of time-series i/o", x.shape, y.shape)
    return x,y

def trim_dataset(mat, batch_size):
    no_rows_drop = mat.shape[0] % batch_size

    if (no_rows_drop > 0):
        return mat[:-no_rows_drop]
    else:
        return mat

BATCH_CURSOR = 0

def next_batch(xvals, yvals):
    global BATCH_CURSOR
    if BATCH_CURSOR >= xvals.shape[0] / BATCH_SIZE:
        BATCH_CURSOR = 0
    pointr = BATCH_CURSOR * BATCH_SIZE
    BATCH_CURSOR = BATCH_CURSOR + 1
    return xvals[pointr : pointr + BATCH_SIZE], yvals[pointr : pointr + BATCH_SIZE]

x_t, y_t = build_timeseries(x_train, 3)
n_steps = TIME_STEPS
n_inputs = 5
n_outputs = 1
n_neurons = 100
n_outputs = 1
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons), output_size = n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_iterations = 10000

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(x_t, y_t)
        sess.run(training_op, feed_dict= {X: X_batch, y: y_batch})
        print(X_batch)
        print(y_batch)
        print(outputs.eval(feed_dict = {X: X_batch, y: y_batch}))
        exit()
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
