import numpy as np
import gym

env = gym.make('CartPole-v0')
obs = env.reset()
print "action size: ", env.action_space.n
print "observation size: ", env.observation_space.shape[0]
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    """
    print obs
    print reward
    print ""
    """
