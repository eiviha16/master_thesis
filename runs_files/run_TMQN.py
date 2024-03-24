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
#config = {"threshold": -1000, "env_name": "cartpole", 'algorithm': 'QTM', "max_update_p": 0.005, "min_update_p": 0.0, 'nr_of_clauses': 940, 'T': int(940 * 0.59), 's': 4.15, 'y_max': 70, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'gamma': 0.981, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.002, 'buffer_size': 6500, 'batch_size': 80, 'epochs': 1, 'test_freq': 25, "save": True, "seed": 42, 'number_of_state_bits_ta': 9, "dataset_file_name": "observation_data"}
#config = {"threshold": -1000, "env_name": "cartpole", 'algorithm': 'QTM', "max_update_p": 0.079, "min_update_p": 0.0, 'nr_of_clauses': 880, 'T': int(880 * 0.89), 's': 4.43, 'y_max': 70, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'gamma': 0.963, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.006, 'buffer_size': 9500, 'batch_size': 32, 'epochs': 7, 'test_freq': 25, "save": False, "seed": 42, 'number_of_state_bits_ta': 4, "dataset_file_name": "observation_data"}
#config = {"env_name": "cartpole", 'batch_size': 32, 'bits_per_feature': 9, 'buffer_size': 9500, 'epochs': 1, 'exploration_p_decay': 0.009000000000000001, 'exploration_p_init': 0.8, 'gamma': 0.944, 'max_update_p': 0.195, 'nr_of_clauses': 920, 'number_of_state_bits_ta': 8, 'specificity': 2.170000000000001, 't': 0.51, 'y_max': 60, 'y_min': 25}
config = {"env_name": "cartpole",'algorithm': 'TMQN', 'nr_of_clauses': 920, 'T': int(920 * 0.13), 'max_update_p': 0.005, 'min_update_p': 0, 's': 6.67, 'y_max': 60, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'gamma': 0.975, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.009, 'buffer_size': 2500, 'threshold': 15, 'batch_size': 48, 'epochs': 1, 'test_freq': 25, 'save': False, 'seed': 42, 'number_of_state_bits_ta': 3, 'dataset_file_name': 'observation_data'}

print(config)
env = gym.make("CartPole-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=5_000)

from test_policy import test_policy
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

#tms = torch.load(f'../results/TM_PPO/{agent.run_id}/best')
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    #eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses
    agent.policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy)
exit(0)
#test_policy(agent.policy)
agent.policy.tm1.set_state()
agent.policy.tm2.set_state()
save_file = f'results/TMQN/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy)
