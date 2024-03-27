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
