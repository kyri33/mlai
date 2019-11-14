import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import collections
import gym
import threading

class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = keras.layers.Dense(100, activation='relu')
        self.policy_logits = keras.layers.Dense(action_size)

        self.dense2 = keras.layers.Dense(100, activation='relu')
        self.values = keras.layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)

        v = self.dense2(inputs)
        values = self.values(v)
        return logits, values

class MasterAgent():
    def __init__(self):
        self.game_name = "CartPole-v0"
        save_dir = args.save_dir
        self.save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_size.n
        self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)
        print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)

class Memory:
    def __init__(self, mem_size):
        self.states = collections.deque(maxlen=mem_size)
        self.actions = collections.deque(maxlen=mem_size)
        self.rewards = collections.deque(maxlen=mem_size)
        self.mem_size = mem_size

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = collections.deque(maxlen=self.mem_size)
        self.actions = collections.deque(maxlen=self.mem_size)
        self.rewards = collections.deque(maxlen=self.mem_size)

class Worker(threading.Thread):
    # global variables
    global_episode = 0
    global_moving_average = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init(self, state_size, action_size, global_model, opt, result_queue, idx, game_name="CartPole-v0", save_dir="/tmp"):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(state_size, action_size)
        self.worder_idx = idx
        self.game_name = game_name
        self.env = gym.make(game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(tf.convert_to_tensor(current_state[None, :],
                        dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == args.update_freq or done:
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done)