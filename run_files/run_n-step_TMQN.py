import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_TMQN import TMQN
from algorithms.policy.RTM import Policy

#config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 17, 'nr_of_clauses': 980, 'T': (980 * 0.34), 's': 8.58, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'gamma': 0.992, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 4000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 6, 'update_grad': 0.05, 'update_freq': 8}

#Winner run 76 - 500.0 - 500.0 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
#run 94 - 498.95 - 7.41 ,config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}
#config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 350, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}
#config = {'algorithm': 'n_step_TMQN', 'n_steps': 17, 'nr_of_clauses': 980, 'T': int(980 * 0.34), 's': 8.58, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'gamma': 0.992, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 4000, 'batch_size': 80, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 6}
#config = {'algorithm': 'n_step_TMQN', 'n_steps': 16, 'nr_of_clauses': 960, 'T': int(960 * 0.86), 's': 5.37, 'y_max': 75, 'y_min': 30, 'max_update_p': 0.05, "min_update_p": 0.0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.987, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 9500, 'batch_size': 96, 'epochs': 6, 'test_freq': 1,  "save": False, "threshold": 0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8, "dataset_file_name": "observation_data"}
#config = {"env_name": "cartpole", 'algorithm': 'n_step_QTM', 'n_steps': 15, 'nr_of_clauses': 1040, 'T': int(1040 * 0.48), 's': 5.84, 'y_max': 75, 'y_min': 25, 'max_update_p': 0.156, "min_update_p": 0.0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'gamma': 0.979, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.009, 'buffer_size': 4000, 'batch_size': 16, 'epochs': 7, 'test_freq': 1,  "save": True, "threshold": 0, 'dynamic_memory': False, 'number_of_state_bits_ta': 5, "dataset_file_name": "observation_data"}
config = {"env_name": "acrobot", 'algorithm': 'n_step_QTM', 'n_steps': 47, 'nr_of_clauses': 1980, 'T': 1841, 'max_update_p': 0.042, 'min_update_p': 0, 's': 2.200000000000001, 'y_max': -10, 'y_min': -70, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, 'gamma': 0.952, 'exploration_prob_init': 0.5, 'exploration_prob_decay': 0.009000000000000001, 'buffer_size': 9500, 'threshold': -495, 'batch_size': 16, 'epochs': 5, 'test_freq': 1, 'save': True, 'seed': 42, 'number_of_state_bits_ta': 9, 'dataset_file_name': 'acrobot_obs_data'}
print(config)

#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")

agent = TMQN(env, Policy, config)
# agent.learn(nr_of_episodes=10000)
agent.learn(nr_of_episodes=1000)

from test_policy import test_policy


#test_policy(agent.policy)
#test_policy(agent.current_policy)
#save_file = f'results/n_step_TMQN/{agent.run_id}/final_test_results'


#save_file = f'results/n_step_TMQN/{agent.run_id}'
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

#tms = torch.load(f'results/n_step_TMQN/{agent.run_id}/best')

agent.policy.tms[0].set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.policy.tms[1].set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])
agent.policy.tms[2].set_params(tms[2]['ta_state'], tms[2]['clause_sign'], tms[2]['clause_output'], tms[2]['feedback_to_clauses'])

test_policy(save_file, agent.policy, config['env_name'])
