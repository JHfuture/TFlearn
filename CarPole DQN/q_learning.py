import tensorflow as tf
import numpy as np
import gym
import random

# Implements DQN with experience replay and delayed target network update
class Q_Learning:
    MAX_STEP_COUNT = 1000
    REG_FACTOR = 0.001
    LEARNING_RATE = 0.0001
    EPISODE_COUNT = 20000
    RANDOM_ACTION_PROB = 0.5
    MIN_RANDOM_ACTION_PROB = 0.01
    EPSILON_DECAY = 0.99
    MAX_EXPERIENCE_LENGTH = 10000
    MINI_BATCH_SIZE = 32
    TARGET_UPDATE_COUNT = 100
    DISCOUNT_FACTOR = 0.9
    def __init__(self, env, nn_config):
        self.env = gym.make(env)
        self.nn_config = nn_config
        self.experience = [] #(s, a, r, s', end)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n
        # initialize the neural network
        self.init_nn()

    def init_nn(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_size]) 
        hidden_count = self.nn_config['hidden_count'] #the size of each hidden layer
        hidden_size = len(hidden_count)
        
        in_size = self.input_size
        in_data = self.x
        self.weights = []
        self.params = []
        
        for hidden_index in range(hidden_size): 
            layer_number = hidden_index + 1
            out_size = hidden_count[hidden_index]
            layer_name = "layer" + str(layer_number)

            with tf.name_scope(layer_name):
                init_w = tf.truncated_normal(shape = [in_size, out_size], stddev = 0.01)
                w = tf.Variable(init_w, name = 'weight')
                init_b = tf.truncated_normal(shape = [out_size], stddev = 0.001)
                b = tf.Variable(init_b, name = "bias")

                self.weights.append(w)
                self.params.append(w)
                self.params.append(b)

                in_data = tf.nn.relu(tf.matmul(in_data, w) + b)

            in_size = out_size

        with tf.name_scope("layer_out"):
            init_w = tf.truncated_normal(shape = [in_size, self.output_size], stddev = 0.01)
            w = tf.Variable(init_w, name = 'weight')
            init_b = tf.truncated_normal(shape = [self.output_size], stddev = 0.001)
            b = tf.Variable(init_b, name = 'bias')

            self.weights.append(w)
            self.params.append(w)
            self.params.append(b)
            
            self.Q = tf.matmul(in_data, w) + b
        
        self.target_Q = tf.placeholder(tf.float32, [None]) # the target of Q network
        self.action_mask = tf.placeholder(tf.float32, [None, self.output_size]) # the actions taken mask

        self.Q_difference = tf.reduce_sum(tf.mul(self.Q, self.action_mask), 1) - self.target_Q
        self.loss = tf.reduce_mean(tf.square(self.Q_difference))

        for w in self.weights:
            self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))

        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        #optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        self.adam = optimizer.minimize(self.loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self):
        # get target network param
        self.total_steps = 0
        for episode_index in range(self.EPISODE_COUNT):
            self.target_params = self.sess.run(self.params)
            print "episode number: ", episode_index + 1, " total_steps: ", self.total_steps
            # initialize episode
            state = self.env.reset()
            step_count = 0
            while step_count < self.MAX_STEP_COUNT:
                step_count += 1
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
#                if done:
#                    reward = -100
                # store them in experience list
                if len(self.experience) >= self.MAX_EXPERIENCE_LENGTH:
                    self.experience.pop(0)
                self.experience.append((state, action, reward, next_state, done))
                state = next_state
                # get batch experience and train
                if len(self.experience) == self.MAX_EXPERIENCE_LENGTH:
                    mini_batch = random.sample(self.experience, self.MINI_BATCH_SIZE)
                    next_states = [e[3] for e in mini_batch]
                    feed = {self.x: next_states}
                    feed.update(zip(self.params, self.target_params))
                    next_Q = self.sess.run(self.Q, feed)
                    next_Q_max = np.max(next_Q, 1)

                    rewards = np.array([e[2] for e in mini_batch])
                    actions = np.array([e[1] for e in mini_batch])
                    states = [e[0] for e in mini_batch]

                    target_Q = rewards + self.DISCOUNT_FACTOR
                    action_mask = np.zeros((self.MINI_BATCH_SIZE, self.output_size))
                    action_mask[np.arange(0,self.MINI_BATCH_SIZE), actions] = 1

                    feed = {self.x : states, self.target_Q : target_Q, self.action_mask: action_mask}

                    _, l, dq, p = self.sess.run([self.adam, self.loss, self.Q_difference, self.params], feed)
                # update target network to current network
                if self.total_steps % self.TARGET_UPDATE_COUNT == 0:
                    self.target_params = self.sess.run(self.params)
                # decay epsilon greedy epsilon
                self.RANDOM_ACTION_PROB *= self.EPSILON_DECAY
                self.RANDOM_ACTION_PROB = max(self.RANDOM_ACTION_PROB, self.MIN_RANDOM_ACTION_PROB)
                if done:
                    print step_count
                    break

if __name__ == '__main__':
    nn_config = {'hidden_count': [16]}
    ql = Q_Learning('CartPole-v0', nn_config)
    ql.train()
    
