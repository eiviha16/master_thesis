import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.Double_TMQN import TMQN
from algorithms.policy.RTM import Policy


#config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1000, 'T': 100, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8, 'update_grad': 0.75}
#config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1000, 'T': 350, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8, 'update_grad': 0.05, 'update_freq': 3}
#Run 71 / Mean reward: 499.05 std: 6.918634258291155 config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 718, 'T': int(718 * 0.8578588678912618), 's': 9.80861195015113, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'gamma': 0.9979953858725568, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 9, 'update_grad': 0.05, 'update_freq': 6}
#config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1160, 'T': int(1160 * 0.31), 's': 9.79, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'gamma': 0.974, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 3000, 'batch_size': 64, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 7, 'update_grad': 0.05, 'update_freq': 2}
#config = {'comment': "Q-value on action", 'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1100, 'T': int(1100 * 0.97), 's': 7.68, 'y_max': 70, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'gamma': 0.967, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 6500, 'batch_size': 48, 'epochs': 2, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 5, 'update_grad': 0, 'update_freq': 9, "dataset_file_name": "observation_data"}
config = {"max_update_p": 0.193, "min_update_p": 0.0, "threshold": -100, "env_name": "cartpole", 'comment': "Q-value on action", 'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1120, 'T': int(1120 * 0.97), 's': 5.8, 'y_max': 70, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'gamma': 0.982, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.001, 'buffer_size': 6500, 'batch_size': 16, 'epochs': 5, 'test_freq': 25, "save": False, "seed": 42, 'dynamic_memory': False, 'number_of_state_bits_ta': 9, 'update_grad': 0, 'update_freq': 8, "dataset_file_name": "observation_data"}

print(config)

env = gym.make("CartPole-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy


tms = torch.load(f'results/Double_TMQN/{agent.run_id}/best')

agent.target_policy.tm1.set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tm2.set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])

save_file = f'results/Double_TMQN/{agent.run_id}'
test_policy(save_file, agent.target_policy)
