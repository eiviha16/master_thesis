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
#config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1000, 'T': 350, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8, 'update_grad': 0.05, 'update_freq': 3}
#Run 71 / Mean reward: 499.05 std: 6.918634258291155 config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 718, 'T': int(718 * 0.8578588678912618), 's': 9.80861195015113, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'gamma': 0.9979953858725568, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 9, 'update_grad': 0.05, 'update_freq': 6}
#config = {'soft_update_type': 'soft_update_2', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1160, 'T': int(1160 * 0.31), 's': 9.79, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'gamma': 0.974, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 3000, 'batch_size': 64, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 7, 'update_grad': 0.05, 'update_freq': 2}
#config = {'comment': "Q-value on action", 'soft_update_type': 'soft_update_1', 'algorithm': 'Double_TMQN', 'nr_of_clauses': 1120, 'T': int(1120 * 0.86), 's': 3.2, 'y_max': 70, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 13, 'gamma': 0.981, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 8500, 'batch_size': 32, 'epochs': 3, 'test_freq': 1, "save": False, "seed": 42,"max_update_p": 0.05, "min_update_p": 0.0, 'dynamic_memory': False, 'number_of_state_bits_ta': 8, 'update_grad': 0.433, "threshold": 100, 'update_freq': -1,"dataset_file_name": "observation_data"}
#config = {'env_name': "acrobot", 'algorithm': 'Double_QTM_a', 'soft_update_type': 'soft_update_1', 'nr_of_clauses': 1240, 'T': 1004, 'max_update_p': 0.157, 'min_update_p': 0, 's': 1.4700000000000004, 'y_max': -5, 'y_min': -70, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'gamma': 0.987, 'exploration_prob_init': 0.8999999999999999, 'exploration_prob_decay': 0.005, 'buffer_size': 4500, 'batch_size': 64, 'epochs': 4, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': -495, 'number_of_state_bits_ta': 8, 'update_grad': 0.893, 'update_freq': -1, 'dataset_file_name': 'acrobot_obs_data'}
#config = {'env_name': "acrobot", 'algorithm': 'Double_QTM_a', 'soft_update_type': 'soft_update_1', 'nr_of_clauses': 1140, 'T': 353, 'max_update_p': 0.025, 'min_update_p': 0, 's': 6.240000000000005, 'y_max': -5, 'y_min': -75, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'gamma': 0.966, 'exploration_prob_init': 0.8999999999999999, 'exploration_prob_decay': 0.007, 'buffer_size': 2500, 'batch_size': 48, 'epochs': 6, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': -495, 'number_of_state_bits_ta': 7, 'update_grad': 0.305, 'update_freq': -1, 'dataset_file_name': 'acrobot_obs_data'}
config = {'env_name': "cartpole", 'algorithm': 'Double_QTM_b',  'soft_update_type': 'soft_update_2', 'nr_of_clauses': 880, 'T': 844, 'max_update_p': 0.101, 'min_update_p': 0, 's': 7.790000000000006, 'y_max': 70, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'gamma': 0.973, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.005, 'buffer_size': 4500, 'batch_size': 16, 'epochs': 6, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': 20, 'number_of_state_bits_ta': 8, 'update_grad': -1, 'update_freq': 1, 'dataset_file_name': 'observation_data'}
print(config)

env = gym.make("CartPole-v1")
#env = gym.make("Acrobot-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy


tms = torch.load(f'results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

agent.target_policy.tm1.set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tm2.set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])
#agent.target_policy.tm2.set_params(tms[2]['ta_state'], tms[2]['clause_sign'], tms[2]['clause_output'], tms[2]['feedback_to_clauses'])

save_file = f'results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}'
test_policy(save_file, agent.target_policy, config["env_name"])

exit(0)

import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.Double_TMQN import TMQN
from algorithms.policy.RTM import Policy


config = {"env_name": "acrobot", 'algorithm': 'Double_QTM_b', 'soft_update_type': 'soft_update_2', 'nr_of_clauses': 1580, 'T': 711, 'max_update_p': 0.055, 'min_update_p': 0, 's': 3.8000000000000025, 'y_max': -5, 'y_min': -55, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, 'gamma': 0.959, 'exploration_prob_init': 0.7, 'exploration_prob_decay': 0.004, 'buffer_size': 9500, 'batch_size': 80, 'epochs': 4, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': -495, 'number_of_state_bits_ta': 7, 'update_grad': -1, 'update_freq': 5, 'dataset_file_name': 'acrobot_obs_data'}

print(config)

#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=500)

from test_policy import test_policy


save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

agent.target_policy.tms[0].set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tms[1].set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])
agent.target_policy.tms[2].set_params(tms[2]['ta_state'], tms[2]['clause_sign'], tms[2]['clause_output'], tms[2]['feedback_to_clauses'])

test_policy(save_file, agent.target_policy)
