import numpy as np
import tensorflow as tf

config = tf.ConfigProto(inter_op_parallelism_threads=4,
                        intra_op_parallelism_threads=4)

x = tf.placeholder(tf.float32, shape=(1,5), name='x')
w = tf.placeholder(tf.float32, shape=(5, 1), name='w')
b = tf.placeholder(tf.float32, shape=(1), name='b')

y = tf.matmul(x, w) + b
s = tf.nn.sigmoid(y)

with tf.Session(config = config) as tfs:
    tfs.run(tf.global_variables_initializer())
    w_t = [[.1, .7, .75, .60, .20]]
    x_1 = [[10, 2, 1, 6, 2]]
    b_1 = [1]
    w_1 = np.transpose(w_t)
    value_1 = tfs.run(s,
        feed_dict = {
            x: x_1,
            w: w_1,
            b: b_1
        }
    )
print('value for threshold calculation', value_1)