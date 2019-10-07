import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("datasets/jse/quotes.csv", sep=";")
# Hello motto
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

print(x_train.shape)
print(x_train.reshape(-1).shape)

scaler = MinMaxScaler()

close_col = 3

smoothing_window_size = 250

for di in range(0, x_train.shape[0], smoothing_window_size):
    scaler.fit(x_train[di:di+smoothing_window_size,:])
    x_train[di:di+smoothing_window_size,:] = scaler.transform(x_train[di:di+smoothing_window_size,:])

# scaler.fit(x_train[di + smoothing_window_size:, :])
# x_train[di + smoothing_window_size:, :] = scaler.transform(x_train[di+smoothing_window_size:, :])
x_test = scaler.transform(x_test)
x_test = x_test[:, close_col]

all_mid_data = np.concatenate([x_train[:, close_col], x_test[:]], axis=0)

# DATA GENERATION OF BATCHES
class DataGeneratorSeq(object):

    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        batch_data = np.zeros((self._batch_size, 5), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size, 1), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                self._cursor[b] = np.random.randint(0, (b + 1)*self._segments)
            
            batch_data[b] = self._prices[self._cursor[b], :]
            batch_labels[b] = self._prices[self._cursor[b]+np.random.randint(1, 5), 3]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length
        
        return batch_data, batch_labels

    def unroll_batches(self):
        
        unroll_data, unroll_labels = [],[]
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)
        return unroll_data, unroll_labels
    
    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b+1)* self._segments, self._prices_length-1))


dg = DataGeneratorSeq(x_train, 5, 2)
u_data, u_labels = dg.unroll_batches()
for ui,(dat, lbl) in enumerate(zip(u_data, u_labels)):
    print("\n\nUnrolled index %d"%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tINputs: ', dat)
    print('\n\tOutput:',lbl)
            

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

for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, D],name='train_inputs_%d'%ui))
    train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size, 1], name='train_outputs_%d'%ui))

lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=num_nodes[li], 
                                    state_is_tuple=True,
                                    initializer=tf.contrib.layers.xavier_initializer())
            for li in range (n_layers)]

drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
    lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout)
    for lstm in lstm_cells]

drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

# Weight and bias vectors for linear regression of output
w = tf.get_variable('w', shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b', initializer = tf.random_uniform([1], -0.1, 0.1))

# Cell state and hidden state variables to maintain state of lstm
c, h = [], []
initial_state = []
for li in range(n_layers):
    c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

# Transform data into shape [num_unrollings, batch_size(timesteps), inputs]

all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)

# All outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.nn.dynamic_rnn(
    drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
    time_major = True, dtype=tf.float32
)

# Reshape outputs
all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * num_unrollings, num_nodes[-1]])

# Apply linear regression
all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)

split_outputs = tf.split(all_outputs, num_unrollings, axis=0)

# Calculating loss, note that error is summed and not averaged ?
print("Defining training Loss")
loss = 0.0
with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)] +
                            [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
    for ui in range(num_unrollings):
        loss += tf.reduce_mean(0.5*(split_outputs[ui] - train_outputs[ui])**2)

print("Learning rate decay operations")
global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.assign(global_step, global_step+1)
tf_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)
tf_min_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)

learning_rate = tf.maximum(
    tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1,decay_rate=0.5,staircase=True),
    tf_min_learning_rate
)

# Optimizer
print('TF Optimization operations')
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(zip(gradients, v))

print("\tAll done")


print('Defining prediction related TF functions')

sample_inputs = tf.placeholder(tf.float32, shape = [1, D]) # ?

sample_c, sample_h, initial_sample_state = [], [], []
for li in range(n_layers):
    sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
    initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li], sample_h[li]))

reset_sample_states = tf.group(*[tf.assign(sample_c[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)], 
                            *[tf.assign(sample_h[li], tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs, 0), 
                                initial_state=tuple(initial_sample_state), time_major = True,
                                dtype=tf.float32)
with tf.control_dependencies([tf.assign(sample_c[li], sample_state[li][0]) for li in range(n_layers)] +
                            [tf.assign(sample_h[li], sample_state[li][0]) for li in range(n_layers)]):
    sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1, -1]), w, b)

print("\tAll Done")

# RUNNING THE LSTM

epochs = 30
valid_summary = 1 # Interval to make test predictions

n_predict_once = 50 # Number of time steps you continuously predict for

train_seq_length = x_train.shape[0] # Full length of training data

train_mse_ot = [] # Training errors acumulated 
test_mse_ot = [] # Test loss
predictions_over_time = [] # Accumulate predictions

session = tf.InteractiveSession()

tf.global_variables_initializer().run()

# Used for decaying learning rate
loss_nondecrease_count = 0
loss_nondecrease_threshold = 2 # If test error hasn't in creased in this many steps then decrease learning rate

print("Initialized")
average_loss = 0

data_gen = DataGeneratorSeq(x_train, batch_size, num_unrollings)

x_axis_seq = []

# Points you start our test predictions from
test_points_seq = np.arange(0, x_test.shape[0], 50).tolist()

for ep in range(epochs):

    # ===================== TRAINING
    for step in range(train_seq_length // batch_size):
        u_data, u_labels = data_gen.unroll_batches()

        feed_dict = {}
        for ui, (dat,lbl) in enumerate(zip(u_data, u_labels)):
            feed_dict[train_inputs[ui]] = dat.reshape(-1, 1) # !!!!! TODO TODO TODO
            feed_dict[train_outputs[ui]] = lbl.reshape(-1, 1)

        feed_dict.update({tf.learning_rate: 0.0001, tf_min_learning_rate: 0.000001})

        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += 1

    # ===================== VALIDATION
    if (ep+1) % valid_summary == 0:

        average_loss = average_loss / (valid_summary * (train_seq_length//batch_size))

        print("Average loss at step %d: %f" % (ep+1, average_loss))

        train_mse_ot.append(average_loss)
        average_loss = 0 # Reset loss ?
        predictions_seq = []
        mse_test_loss_seq = []

        # ===================== UPDATING STATE AND MAKING PREDICTIONS
        for w_i in test_points_seq:
            mse_test_loss = 0.0
            our_predictions = []

            if (ep+1) - valid_summary == 0:
                x_axis = [] # Only calculate x axis values in first validation epoch
            
            # Feed in the recent past behavior of stock prices
            # to make predictions form that point onwards
            for tr_i in range(w_i - num_unrollings+1, w_i-1):
                current_price
