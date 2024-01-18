import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.Double_TMQN import TMQN
from algorithms.policy.RTM import Policy
#adjusting memory and batch size can change things
#config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1000, 'T': 100, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8, 'update_grad': 0.75}
config = {'soft_update_type': 'soft_update_1', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1000, 'T': 100, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8, 'update_grad': 0.75, 'update_freq': 1}
print(config)

env = gym.make("CartPole-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=10000)

from test_policy import test_policy

#test_policy(agent.policy)
agent.target_policy.tm1.set_state()
agent.target_policy.tm2.set_state()
test_policy(agent.target_policy)
