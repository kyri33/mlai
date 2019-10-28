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

bandos = Bandits()

W = tf.Variable(initial_value=tf.ones([bandos.num_bandits, bandos.action_size]),
    shape=[bandos.num_bandits, bandos.action_size])
b = tf.zeros(shape=[bandos.action_size])
#output = tf.Variable(initial_value = tf.ones([1, 4]), dtype=tf.float32)
output = None

@tf.function
def pickAction(state_in):
    global output
    state_in_oh = tf.one_hot(state_in, bandos.num_bandits)
    state_in_OH = tf.reshape(state_in_oh, [1, -1])
    output = tf.nn.sigmoid(tf.matmul(state_in_OH, W) + b)
    #output = tf.reshape(o, [-1])
    act = tf.argmax(output, 1, output_type=tf.dtypes.int32)
    return act

@tf.function
def train_act(state_in, reward, action):
    global output
    def loss():
        respons = tf.slice(tf.reshape(output, [-1]), action, [1])
        return -(tf.math.log(respons) * reward)
    optimizer = tf.optimizers.SGD()
    optimizer.minimize(loss, var_list=[W], grad_loss=None, name=None)

lr = 0.001
e = 0.2

iter = 10000
total_score = np.zeros([bandos.num_bandits, bandos.action_size])

i = 0
while i < iter:
    state = bandos.pickBandit()
    ran = np.random.rand(0)

    if ran < e:
        action = np.random.randint(0, bandos.action_size)
    else:
        action = pickAction(state)
    reward = bandos.pullBandit(state, action)
    train_act(state, reward, action)

    i += 1

