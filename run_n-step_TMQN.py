import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_TMQN import TMQN
from algorithms.policy.RTM import Policy

# config = {'algorithm': 'n_step_TMQN', 'nr_of_clauses': 1000, 'T': 100, 's': 3.8, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.80, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
# 492, 14 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 100, 's': 3.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.80, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
# run_5: 3310 - 497.42 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 100, 's': 5.0, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
# run 6: 3244 - 417.87 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 100, 's': 4.0, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
# run 7: 2035 - 482.07 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 100, 's': 4.5, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
# run 10: 8374 - 499.98 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 100, 's': 5.1, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
# run 12: 9537 - 500.0 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 100, 's': 5.2, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
# run 70> / 500.0 / config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 5.2, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
#Winner run 76 - 500.0 - 500.0 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
# agent.learn(nr_of_episodes=10000)
agent.learn(nr_of_episodes=10000)

from test_policy import test_policy

test_policy(agent.policy)
#test_policy(agent.current_policy)
agent.policy.tm1.set_state()
agent.policy.tm2.set_state()
test_policy(agent.policy)
