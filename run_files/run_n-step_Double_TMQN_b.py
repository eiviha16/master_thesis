import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_Double_TMQN import TMQN
from algorithms.policy.RTM import Policy

#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 19, 'nr_of_clauses': 980, 'T': (980 * 0.51), 's': 6.56, 'y_max': 60, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'gamma': 0.977, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 8000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 5, 'update_grad': 0.05, 'update_freq': 7}
#config = {"env_name": "cartpole", 'comment': "Q-value on action", 'algorithm': 'n_step_Double_QTM', 'soft_update_type': 'soft_update_2', 'n_steps': 29, 'max_update_p': 0.131, "min_update_p": 0.0,'nr_of_clauses': 1020, 'T': (1020 * 0.5), 's': 3.59, 'y_max': 70, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, 'gamma': 0.978, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.001, 'buffer_size': 2500, 'batch_size': 16, 'epochs': 3, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 9, 'update_freq': 7,  "dataset_file_name": "observation_data", "threshold": 0}# "observation_data"}
#config = {"env_name": "cartpole", 'comment': "Q-value on action", 'algorithm': 'n_step_Double_QTM', 'soft_update_type': 'soft_update_2', 'n_steps': 38, 'max_update_p': 0.185, "min_update_p": 0.0,'nr_of_clauses': 960, 'T': (960 * 0.5), 's': 4.83, 'y_max': 75, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'gamma': 0.973, 'exploration_prob_init': 0.005, 'exploration_prob_decay': 0.005, 'buffer_size': 2000, 'batch_size': 48, 'epochs': 2, 'test_freq': 50,  "save": True, 'number_of_state_bits_ta': 9, 'update_freq': 4,  "dataset_file_name": "observation_data", "threshold": 0}# "observation_data"}
#config = {"env_name": "cartpole", 'comment': "Q-value on action", 'algorithm': 'n_step_Double_QTM', 'soft_update_type': 'soft_update_2', 'n_steps': 42, 'max_update_p': 0.18, "min_update_p": 0.0,'nr_of_clauses': 980, 'T': (980 * 0.36), 's': 2.97, 'y_max': 60, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'gamma': 0.979, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.005, 'buffer_size': 9500, 'batch_size': 32, 'epochs': 6, 'test_freq': 25,  "save": True, 'number_of_state_bits_ta': 4, 'update_freq': 1,  "dataset_file_name": "observation_data", "threshold": 0}# "observation_data"}
#config = {"env_name": "cartpole", 'comment': "Q-value on action", 'algorithm': 'n_step_Double_QTM', 'soft_update_type': 'soft_update_2', 'n_steps': 8, 'max_update_p': 0.021, "min_update_p": 0.0,'nr_of_clauses': 1000, 'T': (1000 * 0.22), 's': 4.35, 'y_max': 60, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'gamma': 0.981, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.007, 'buffer_size': 5000, 'batch_size': 80, 'epochs': 5, 'test_freq': 25,  "save": False, 'number_of_state_bits_ta': 5, 'update_freq': 1,  "dataset_file_name": "observation_data", "threshold": 0}# "observation_data"}
#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': 19, 'nr_of_clauses': 1080, 'T': (1080 * 0.64), 's': 9.92, 'y_max': 70, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.966, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 500, 'batch_size': 96, 'epochs': 4, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 8, 'update_grad': 0.129, 'update_freq': 9999999}
#config = {"env_name": "cartpole", 'algorithm': 'n_step_Double_QTM', 'soft_update_type': 'soft_update_2', 'max_update_p': 0.179, "min_update_p": 0.0, 'n_steps': 6, 'nr_of_clauses': 1100, 'T': (1100 * 0.05), 's': 6.04, 'y_max': 65, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'gamma': 0.975, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.007, 'buffer_size': 2000, 'batch_size': 96, 'epochs': 6, 'test_freq': 25,  "save": False, 'number_of_state_bits_ta': 7,  'update_freq': 6, "dataset_file_name": "observation_data", "threshold": 0}
#config = {"env_name": "cartpole", 'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 17, 'nr_of_clauses': 900, 'T': 756, 'max_update_p': 0.153, 'min_update_p': 0, 's': 5.8300000000000045, 'y_max': 60, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'gamma': 0.959, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.002, 'buffer_size': 1000, 'batch_size': 112, 'epochs': 7, 'test_freq': 25, 'save': False, 'seed': 42, 'threshold': 15, 'number_of_state_bits_ta': 9, 'update_grad': -1, 'update_freq': 7, 'dataset_file_name': 'observation_data'}
#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 25, 'nr_of_clauses': 920, 'T': int(920 * 0.3), 'max_update_p': 0.075, 'min_update_p': 0, 's': 1.99, 'y_max': 65, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'gamma': 0.933, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.001, 'buffer_size': 6500, 'batch_size': 64, 'epochs': 2, 'test_freq': 25, 'save': False, 'seed': 42, 'threshold': 15, 'number_of_state_bits_ta': 3, 'update_freq': 2, 'dataset_file_name': 'observation_data'}
#config = {"env_name": "cartpole", 'algorithm': 'n_step_Double_QTM_b', 'soft_update_type': 'soft_update_2', 'n_steps': 6, 'nr_of_clauses': 1180, 'T': int(1180 * 0.79), 'max_update_p': 0.152, 'min_update_p': 0, 's': 5.19, 'y_max': 60, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.963, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.006, 'buffer_size': 4000, 'batch_size': 16, 'epochs': 1, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': 15, 'number_of_state_bits_ta': 3, 'update_freq': 4, 'dataset_file_name': 'observation_data'}
config = {"env_name": "acrobot", 'algorithm': 'n_step_Double_QTM_b', 'soft_update_type': 'soft_update_2', 'n_steps': 14, 'nr_of_clauses': 1860, 'T': 613, 'max_update_p': 0.151, 'min_update_p': 0, 's': 3.6000000000000023, 'y_max': -10, 'y_min': -70, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'gamma': 0.93, 'exploration_prob_init': 0.7, 'exploration_prob_decay': 0.008, 'buffer_size': 4000, 'batch_size': 48, 'epochs': 1, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': -495, 'number_of_state_bits_ta': 6, 'update_grad': -1, 'update_freq': 3, 'dataset_file_name': 'acrobot_obs_data'}
#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")

agent = TMQN(env, Policy, config)
#agent.learn(nr_of_episodes=500)

from test_policy import test_policy
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/run_2/final_test_results'
#save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
#tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/run_2/best')

#save_file = f'results/n_step_Double_TMQN/{agent.run_id}/final_test_results'
#tms = torch.load(f'results/n_step_Double_TMQN/{agent.run_id}/best')


agent.target_policy.tms[0].set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tms[1].set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])
agent.target_policy.tms[2].set_params(tms[2]['ta_state'], tms[2]['clause_sign'], tms[2]['clause_output'], tms[2]['feedback_to_clauses'])

test_policy(save_file, agent.target_policy, config['env_name'])
