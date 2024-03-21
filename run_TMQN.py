import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.TMQN import TMQN
from algorithms.policy.RTM import Policy

#run 75: 496.11 - config = {'algorithm': 'TMQN', 'nr_of_clauses': 1000, 'T': 100, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}
#config = {'algorithm': 'TMQN', 'nr_of_clauses': 1000, 'T': 350, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8}
#config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1160, 'T': int(1160 * 0.31), 's': 9.79, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'gamma': 0.974, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 1160, 'batch_size': 64, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 7, 'update_grad': 0.05, 'update_freq': 2}

#config = {'algorithm': 'TMQN', 'nr_of_clauses': 1160, 'T': int(1160 * 0.31), 's': 9.79, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'gamma': 0.974, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 3000, 'batch_size': 64, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, 'dynamic_memory': False, 'number_of_state_bits_ta': 7}
#config = {'algorithm': 'TMQN', 'nr_of_clauses': 1120, 'T': int(1120 * 0.67), 's': 8.86, 'y_max': 60, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'gamma': 0.979, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 9000, 'batch_size': 32, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 6}
config = {'algorithm': 'TMQN', "max_update_p": 0.05, "min_update_p": 0.0, 'nr_of_clauses': 1040, 'T': int(1040 * 0.26), 's': 8.9, 'y_max': 75, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 13, 'gamma': 0.992, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 5500, 'batch_size': 112, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 8, "dataset_file_name": "observation_data"}
print(config)
env = gym.make("CartPole-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

#test_policy(agent.policy)
agent.policy.tm1.set_state()
agent.policy.tm2.set_state()
save_file = f'results/TMQN/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy)
