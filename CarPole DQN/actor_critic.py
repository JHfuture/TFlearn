import tensorflow as tf
import numpy as np
import random
import gym

class Actor_Critic:
    POLICY_LEARNING_RATE = 0.003
    VALUE_LEARNING_RATE = 0.1
    DISCOUNT_VALUE = 0.97
    MAX_STEP_COUNT = 200

    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.init_nn()
        self.sess = tf.Session()
        self.init = tf.initialize_all_variables()
        self.sess.run(self.init)

    def init_nn(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.action_mask = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.mc_value = tf.placeholder(tf.float32, [None])

        # policy network
        with tf.name_scope('policy_network'):
            w1 = tf.Variable(tf.truncated_normal(shape = [self.state_size, 16], stddev = 0.01), name = 'w1')
            b1 = tf.Variable(tf.truncated_normal(shape = [16], stddev = 0.001), name = 'b1')
            h1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)
            w2 = tf.Variable(tf.truncated_normal(shape = [16, 16], stddev = 0.01), name = 'w2')
            b2 = tf.Variable(tf.truncated_normal(shape = [16], stddev = 0.001), name = 'b2')
            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
            wout = tf.Variable(tf.truncated_normal(shape = [16, self.action_size], stddev = 0.01), name = 'wout')
            bout = tf.Variable(tf.truncated_normal(shape = [self.action_size], stddev = 0.001), name = 'bout')
            hout = tf.nn.softmax(tf.matmul(h2, wout) + bout)
            
            # the output action
            self.probability = hout
            loss = -tf.reduce_sum(tf.log(tf.reduce_sum(tf.mul(hout, self.action_mask), 1)) * self.advantage, 0)
            optimizer = tf.train.AdamOptimizer(self.POLICY_LEARNING_RATE)
            self.actor_train_op = optimizer.minimize(loss)

        with tf.name_scope('value_network'):
            w1 = tf.Variable(tf.truncated_normal(shape = [self.state_size, 16], stddev = 0.01), name = 'w1')
            b1 = tf.Variable(tf.truncated_normal(shape = [16], stddev = 0.001), name = 'b1')
            h1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)
            w2 = tf.Variable(tf.truncated_normal(shape = [16, 16], stddev = 0.01), name = 'w2')
            b2 = tf.Variable(tf.truncated_normal(shape = [16], stddev = 0.001), name = 'b2')
            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
            wout = tf.Variable(tf.truncated_normal(shape = [16, 1], stddev = 0.01), name = 'wout')
            bout = tf.Variable(tf.truncated_normal(shape = [1], stddev = 0.001), name = 'bout')
            self.value = tf.reduce_sum(tf.matmul(h2, wout) + bout, 1)
            
            # to optimize value function
            loss = tf.reduce_sum(tf.square(self.mc_value - self.value), 0)
            optimizer = tf.train.AdamOptimizer(self.VALUE_LEARNING_RATE)
            self.critic_train_op = optimizer.minimize(loss)

    def train(self):
        # train for one single episode

        state = np.array([], dtype = np.float32).reshape(0, self.state_size)
        action_mask = np.array([], dtype = np.float32).reshape(0, self.action_size)
        reward = np.array([], dtype = np.float32).reshape(0)
        advantage = np.array([], dtype = np.float32).reshape(0)
        mc_value = np.array([], dtype = np.float32).reshape(0)

        # get the episode by policy network
        s = self.env.reset()
        for _ in range(self.MAX_STEP_COUNT):
            state = np.vstack((state, s))
            probability = self.sess.run(self.probability, {self.state: np.expand_dims(s, 0)})[0]
            a = 0 if random.uniform(0,1) < probability[0] else 1
            s, r, d, _ = self.env.step(a)

            a_mask = np.zeros(self.action_size, dtype = np.float32)
            a_mask[a] = 1.
            action_mask = np.vstack((action_mask, a_mask))

            reward = np.hstack((reward, r))

            if d:
                break

        # calculate the MC_value and advantage
        r = -100.
        vs2= 0.
        vs1 = 0.
        for step_index in range(len(reward)-1, -1, -1):
            r = self.DISCOUNT_VALUE * r + reward[step_index]
            s = state[step_index]
            mc_value = np.hstack((r, mc_value))

            vs1 = self.sess.run(self.value, {self.state: np.expand_dims(s,0)})[0]
            adv = reward[step_index] + self.DISCOUNT_VALUE * vs2 - vs1
            advantage = np.hstack((adv, advantage))

            vs2 = vs1

        feed = {self.state: state, self.mc_value: mc_value}
        _ = self.sess.run(self.critic_train_op, feed)

        feed = {self.state: state, self.action_mask: action_mask, self.advantage: advantage}
        _ = self.sess.run(self.actor_train_op, feed)

        # return the step count of this episodde
        return len(reward)

    def test(self):
        s = self.env.reset()
        step_count = 0
        for _ in range(self.MAX_STEP_COUNT):
            step_count += 1
            probability = self.sess.run(self.probability, {self.state: np.expand_dims(s, 0)})[0]
            a = 0 if random.uniform(0,1) < probability[0] else 1
            s, r, d, _ = self.env.step(a)
            if d:
                break
        return step_count



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ac = Actor_Critic(env)
    total_episode_count = 1000
    total_test_count = 1000
    for episode_count in range(total_episode_count):
        print "episode #", episode_count + 1
        print ac.train()
    total_step = 0.
    for episode_count in range(total_test_count):
        total_step += ac.test()
    print "=======test case======"
    print "go through ", total_test_count, " episodes, average step: ", total_step / total_test_count

