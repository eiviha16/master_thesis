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
#config = {'comment': "Q-value on action",'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': 30, 'max_update_p': 0.05, "min_update_p": 0.0, 'nr_of_clauses': 3040, 'T': (3040 * 0.84), 's': 9.0, 'y_max': -50, 'y_min': -100, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'gamma': 0.977, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 9000, 'batch_size': 16, 'epochs': 5, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 7, 'update_grad': 0.62, 'update_freq': 100, "dataset_file_name": "acrobot_obs_data"}#"observation_data"}
#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_1', 'n_steps': 19, 'nr_of_clauses': 1080, 'T': (1080 * 0.64), 's': 9.92, 'y_max': 70, 'y_min': 20, 'max_update_p': 0.05, "min_update_p": 0.0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.966, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 500, 'batch_size': 96, 'epochs': 4, 'test_freq': 1,  "save": False, 'number_of_state_bits_ta': 8, 'update_grad': 0.129, 'update_freq': 9999999, "dataset_file_name": "observation_data"}
#config = {"threshold": -500, "env_name": "acrobot", 'algorithm': 'n_step_Double_QTM_a', 'soft_update_type': 'soft_update_1', 'n_steps': 49, 'nr_of_clauses': 1620, 'T': (1620 * 0.56), 's': 2.58, 'y_max': -25, 'y_min': -80, 'max_update_p': 0.145, "min_update_p": 0.0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'gamma': 0.986, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.009, 'buffer_size': 8500, 'batch_size': 64, 'epochs': 6, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 8, 'update_grad': 0.52, 'update_freq': -1, "dataset_file_name": "acrobot_obs_data"}
config = {"env_name": "cartpole", 'algorithm': 'n_step_Double_QTM_a', 'soft_update_type': 'soft_update_1', 'n_steps': 12, 'nr_of_clauses': 1100, 'T': 1089, 'max_update_p': 0.139, 'min_update_p': 0, 's': 9.130000000000008, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'gamma': 0.957, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.002, 'buffer_size': 1000, 'batch_size': 64, 'epochs': 7, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': 20, 'number_of_state_bits_ta': 5, 'update_grad': 0.173, 'update_freq': -1, 'dataset_file_name': 'observation_data'}

#env = gym.make("Acrobot-v1")
env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy


save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

agent.target_policy.tms[0].set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tms[1].set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])
#agent.target_policy.tms[2].set_params(tms[2]['ta_state'], tms[2]['clause_sign'], tms[2]['clause_output'], tms[2]['feedback_to_clauses'])

test_policy(save_file, agent.target_policy, config['env_name'])
