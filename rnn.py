import tensorflow as tf

n_inputs = 3
n_neurons = 5

# 3 inputs with 5 neurons in hidden layers = 5 x 3 weights

# the None is for batching, allows for any number of rows
X0 = tf.placeholder(tf.float32, [None, n_inputs]
# I think this is the looparound input
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b
Y1
