import tensorflow as tf
import numpy as np
import gym
import random

# Implements DQN with experience replay and delayed target network update
class Q_Learning:
    MAX_STEP_COUNT = 200
    REG_FACTOR = 0.005
    LEARNING_RATE = 0.0001
    EPISODE_COUNT = 1000000
    RANDOM_ACTION_PROB = 0.5
    EPSILON_DECAY = 0.9999
    MAX_EXPERIENCE_LENGTH = 500000
    MINI_BATCH_SIZE = 64
    TARGET_UPDATE_COUNT = 100
    DISCOUNT_FACTOR = 0.9
    #TAU = 0.01
    def __init__(self, env, nn_config):
        self.env = gym.make(env)
        self.nn_config = nn_config
        self.experience = [] #(s, a, r, s', end)
        self.input_size = self.env.observation_space.shape[0] #state size
        self.output_size = self.env.action_space.n #action size
        # initialize the neural network
        self.init_nn()

    def init_nn(self):
        # s, a, r, s'
        self.x = tf.placeholder(tf.float32, [None, self.input_size]) 
        self.next_x = tf.placeholder(tf.float32, [None, self.input_size])
        self.action_mask = tf.placeholder(tf.float32, [None, self.output_size])
        self.next_Q = tf.placeholder(tf.float32, [None])

        hidden_count = self.nn_config['hidden_count'] #the size of each hidden layer
        hidden_size = len(hidden_count)
        
        in_size = self.input_size
        in_data = self.x

        # for regularization
        self.Q_network_weights = []

        # for target network updates
        self.Q_network_params = []
        self.target_network_params = []
        self.assign_op = []

        with tf.name_scope("Q_network"):
            for hidden_index in range(hidden_size): 
                layer_number = hidden_index + 1
                out_size = hidden_count[hidden_index]
                layer_name = "layer" + str(layer_number)

                with tf.name_scope(layer_name):
                    init_w = tf.truncated_normal(shape = [in_size, out_size], stddev = 0.01)
                    w = tf.Variable(init_w, name = 'weight')
                    init_b = tf.truncated_normal(shape = [out_size], stddev = 0.001)
                    b = tf.Variable(init_b, name = "bias")

                    self.Q_network_weights.append(w)
                    self.Q_network_params.append(w)
                    self.Q_network_params.append(b)

                    in_data = tf.nn.relu(tf.matmul(in_data, w) + b)

                in_size = out_size

            with tf.name_scope("layer_out"):
                init_w = tf.truncated_normal(shape = [in_size, self.output_size], stddev = 0.01)
                w = tf.Variable(init_w, name = 'weight')
                init_b = tf.truncated_normal(shape = [self.output_size], stddev = 0.001)
                b = tf.Variable(init_b, name = 'bias')

                self.Q_network_weights.append(w)
                self.Q_network_params.append(w)
                self.Q_network_params.append(b)
                
                self.Q = tf.matmul(in_data, w) + b

        in_data = self.next_x
        in_size = self.input_size
        with tf.name_scope("target_network"):

            for hidden_index in range(hidden_size): 
                layer_number = hidden_index + 1
                out_size = hidden_count[hidden_index]
                layer_name = "layer" + str(layer_number)

                with tf.name_scope(layer_name):
                    init_w = tf.truncated_normal(shape = [in_size, out_size], stddev = 0.01)
                    w = tf.Variable(init_w, name = 'weight')
                    init_b = tf.truncated_normal(shape = [out_size], stddev = 0.001)
                    b = tf.Variable(init_b, name = "bias")

                    self.target_network_params.append(w)
                    self.target_network_params.append(b)

                    in_data = tf.nn.relu(tf.matmul(in_data, w) + b)

                in_size = out_size

            with tf.name_scope("layer_out"):
                init_w = tf.truncated_normal(shape = [in_size, self.output_size], stddev = 0.01)
                w = tf.Variable(init_w, name = 'weight')
                init_b = tf.truncated_normal(shape = [self.output_size], stddev = 0.001)
                b = tf.Variable(init_b, name = 'bias')

                self.target_network_params.append(w)
                self.target_network_params.append(b)
                
                self.target_Q = tf.matmul(in_data, w) + b

        # assign operations
        for param_index in range(len(self.Q_network_params)):
            #updated_param = self.TAU * self.target_network_params[param_index] + (1 - self.TAU * self.Q_network_params[param_index])
            self.assign_op.append(self.target_network_params[param_index].assign(self.Q_network_params[param_index]))
            #self.assign_op.append(self.target_network_params[param_index].assign(updated_param))

        self.loss = tf.reduce_mean(tf.square(tf.reduce_sum(tf.mul(self.Q, self.action_mask),1) - self.next_Q))

        for w in self.Q_network_weights:
            self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))

        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        #optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        self.adam_op = optimizer.minimize(self.loss, var_list = self.Q_network_params)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self):
        # get target network param
        self.total_steps = 0
        for episode_index in range(self.EPISODE_COUNT):
            print "episode number: ", episode_index + 1, " ,total_steps: ", self.total_steps
            # initialize episode
            state = self.env.reset()
            self.step_count = 0
            while self.step_count < self.MAX_STEP_COUNT:
                self.step_count += 1
                self.total_steps += 1
                # epsilon greedy find action
                action = 0
                if len(self.experience) < self.MAX_EXPERIENCE_LENGTH:
                    print "not full of experience replay yet"
                    action = self.env.action_space.sample()
                else:
                    if random.random() < self.RANDOM_ACTION_PROB:
                        action = self.env.action_space.sample()
                    else:
                        feed = {self.x: [state]}
                        Q = self.sess.run(self.Q, feed)
    #                    print "Q: ", Q
                        action = np.argmax(Q, 1)[0]
    #                    print "action: ", action
    #                    print ""
                # get experience (s, a, r, s', end)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -100
                # store them in experience list
                if len(self.experience) >= self.MAX_EXPERIENCE_LENGTH:
                    self.experience.pop(0)
                self.experience.append((state, action, reward, next_state, done))
                state = next_state
                # get batch experience and train
                if len(self.experience) == self.MAX_EXPERIENCE_LENGTH:
                    mini_batch = random.sample(self.experience, self.MINI_BATCH_SIZE)
                    x = [e[0] for e in mini_batch]
                    next_x = [e[3] for e in mini_batch]
                    d = [e[4] for e in mini_batch]
                    feed = {self.x: x, self.next_x: next_x}
                    target_Q = self.sess.run(self.target_Q, feed)

                    rewards = np.array([e[2] for e in mini_batch])
                    actions = np.array([e[1] for e in mini_batch])
                    next_Q = np.max(target_Q, 1) * self.DISCOUNT_FACTOR + rewards
                    action_mask = np.zeros((self.MINI_BATCH_SIZE, self.output_size))
                    action_mask[np.arange(self.MINI_BATCH_SIZE), actions] = 1
                    
                    feed = {self.x: x, self.next_Q: next_Q, self.action_mask: action_mask}
                    _ = self.sess.run(self.adam_op, feed)
                # update target network to current network
                if self.total_steps % self.TARGET_UPDATE_COUNT == 0:
                    _ = self.sess.run(self.assign_op)
                # decay epsilon greedy epsilon
                self.RANDOM_ACTION_PROB *= self.EPSILON_DECAY
                if done:
                    print self.step_count
                    break

if __name__ == '__main__':
    nn_config = {'hidden_count': [128,128]}
    ql = Q_Learning('CartPole-v0', nn_config)
    ql.train()
    
