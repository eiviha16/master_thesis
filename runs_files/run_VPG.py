import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.VPG.VPG import VPG
from algorithms.policy.DNN import ActorPolicy as Policy

config = {'algorithm': 'VPG', 'gamma': 0.98, 'batch_size': 64, 'epochs': 3, 'hidden_size': 32, 'learning_rate': 0.015, 'test_freq': 1,
          "save": True}

print(config)

env = gym.make("CartPole-v1")

agent = VPG(env, Policy, config)
agent.learn(nr_of_episodes=10000)
