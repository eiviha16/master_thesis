import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'number_of_state_bits_ta': 8}


print(config)

env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=10000)

from test_policy import test_policy

agent.policy.actor.tms[0].set_state()
agent.policy.actor.tms[1].set_state()
test_policy(agent.policy.actor)