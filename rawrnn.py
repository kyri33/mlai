import tensorflow as tf

n_inputs = 3
n_neurons = 5

# 3 inputs with 5 neurons in hidden layers = 5 x 3 weights

# the None is for batching, allows for any number of rows
X0 = tf.placeholder(tf.float32, [None, n_inputs]) # t0
X1 = tf.placeholder(tf.float32, [None, n_inputs]) # t1

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b) # t0 output
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b) # t1 output, t0 is added to it

init = tf.global_variables_initializer()

import numpy as np

X0_batch = np.array([[0, 1, 2], [3,4,5],[6,7,8],[9,0,1]])   # t0
X1_batch = np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]])      # t1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})

# STATIC UNROLLING THROUGH TIME USING AN RNN CELL

tf.reset_default_graph()

X0 = tf.placeholder(tf.float32, [None, n_inputs]) # t0
X1 = tf.placeholder(tf.float32, [None, n_inputs]) # t1

# adds the data in the format of [t0, t1, ..., tn]

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
Y0,Y1 = output_seqs

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict = {X0: X0_batch, X1: X1_batch})

# STATIC UNROLLING WITH 3D ARRAY

# array format is [batch size, n_steps (t), n_inputs]
# we then transpose the array so that n_steps comes first and unstack it into individual tensors [t0, t1, ..] like above
# we then stack it for the outputs

tf.reset_default_graph()

n_steps = 2

X_batch = np.array([
    #t = 0      t = 1
    [[0, 1, 2], [9, 8, 7]], # instance 0 
    [[3, 4, 5], [0, 0, 0]], # instance 1 
    [[6, 7, 8], [6, 5, 4]], # instance 2 
    [[9, 0, 1], [3, 2, 1]], # instance 3
])

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output_val = outputs.eval(feed_dict = {X: X_batch})
    print(output_val)
