import tensorflow as tf
import numpy as np

bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)

def pullBandit(band):
    rand = np.random.randn(1)
    if rand > band:
        return 1
    else:
        return -1

tf.compat.v1.reset_default_graph()

W = tf.Variable(tf.ones([num_bandits]))

chosen_action = tf.argmax(W,0)

reward_holder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1])
action_holder = tf.compat.v1.placeholder(dtype=tf.int32, shape=[1])
responsible_action = tf.slice(W, action_holder, [1])
loss = -(tf.math.log(responsible_action) * reward_holder)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001)
update = optimizer.minimize(loss)

num_steps = 1000

e = 0.3 # Chance of picking a random action

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    init.run()
    i = 0
    while i < num_steps:
        if np.random.rand(1) < e:
            act = np.random.randint(num_bandits)
        else:
            act = sess.run(chosen_action)
        reward = pullBandit(bandits[act])
        _,rw,w = sess.run([update, responsible_action, W], feed_dict = {
            reward_holder: [reward],
            action_holder: [act]
        })
        if i % 100 == 0:
            print(w)
        i+=1
    print(w)

print("Done")