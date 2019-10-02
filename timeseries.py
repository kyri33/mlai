import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts = False):
        # Random starting point for batch ? 
        random_start = np.random.rand(batch_size, 1)

        # Total number of time-series samples will always be total - time steps
        ts_start = random_start * (self.xmax - self.xmin-(steps * self.resolution))

        batch_ts = ts_start + np.arange(0.0, steps + 1) * self.resolution #arrange a batch of size steps + 1 from ts_start

        y_batch = np.sin(batch_ts)

        # Formatting for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1, steps, 1), batch_ts
            #return y_batch, batch_ts
        else:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


ts_data = TimeSeriesData(250, 0, 10)

num_time_steps = 30
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.001
num_iter = 5000
batch_size = 1

x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units = num_neurons, activation = tf.nn.relu), output_size = num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()