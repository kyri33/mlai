# NEURAL NETWORK WITH PLAIN TENSORFLOW

import tensorflow as tf

n_inputs = 28 * 28 # MNIST
n_hidden1 = 300 
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None, 1), name="y")

# Manually creating a neuron layer

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = X.get_shape()[1]
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev) # initialization of weights
        W = tf.Variable(init, name="Weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

# USE, uncomment to see but works the same as fully_connected
#with tf.name_scope("dnn"):
#    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
#    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
#    logits = neuron_layer(hidden2, n_outputs, "output", activation=None)

# Now with tf fully_connected function which is same thing as neuron_layer boyaka

from tensorflow.contrib.layers import fully_connected

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

# Now defining the loss function research softmax cross entropy
with tf.name_scope("loss"): # Used before the activation function on the outputs
    # Equivalent to applying softmax activation and then calculating error
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# Now we train a gradient descent optimizer
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
