import tensorflow as tf
import numpy as np

class Bandits:
    def __init__(self):
        self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.state = 0
        self.action_size = self.bandits.shape[1]

    def pickBandit(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state
    
    def pullBandit(self, state, action):
        band = self.bandits[state, action]
        ra = np.random.randn(1)
        if ra > band:
            return 1.0
        else:
            return -1.0
        

class agent:
    def __init__(self, state_size, action_size, lr):
        self.state_in = tf.compat.v1.placeholder(tf.float32, shape=[1])
        self.state_in_OH = tf.one_hot(tf.dtypes.cast(self.state_in, tf.int32), state_size)
        self.w = tf.Variable(initial_value = tf.ones([state_size, action_size]),
                                shape=[state_size, action_size])
        self.b = tf.zeros(shape=[action_size])
        self.o = tf.nn.sigmoid(tf.matmul(self.state_in_OH, self.w) + self.b)
        self.output = tf.reshape(self.o, [-1])

        self.chosen_action = tf.arg_max(self.output, 0)

        self.reward_holder = tf.compat.v1.placeholder(tf.float32, shape=[1])
        self.action_holder = tf.compat.v1.placeholder(tf.int32, shape=[1])
        self.responsible_output = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_output) * self.reward_holder)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
        self.update = self.optimizer.minimize(self.loss)

tf.reset_default_graph()

lr = 0.001
e = 0.2

bandos = Bandits()
james = agent(bandos.num_bandits, bandos.action_size, lr)

num_iter = 10000
total_score = np.zeros([bandos.num_bandits, bandos.action_size])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    i = 0
    while i < num_iter:
        state = bandos.pickBandit()
        ran = np.random.rand(0)
        if ran < e:
            action = float(np.random.randint(0, bandos.action_size))
        else:
            action = sess.run(james.chosen_action, feed_dict = {james.state_in: [float(state)]})
        
        reward = bandos.pullBandit(state, action)

        feed_dict = {
            james.reward_holder: [reward],
            james.action_holder: [action],
            james.state_in: [float(state)]
        }

        _,ww = sess.run([james.update, james.w], feed_dict = feed_dict)

        total_score[state, action] += reward

        if i % 200 == 0:
            print("Mean Reward:", np.mean(total_score, axis=1))
            print(ww)
        i += 1