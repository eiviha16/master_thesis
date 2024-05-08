import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_Double_QTM import TMQN
from algorithms.policy.RTM import Policy

config = {"env_name": "cartpole", 'algorithm': 'n_step_Double_QTM_a', 'soft_update_type': 'soft_update_1', 'n_steps': 12, 'nr_of_clauses': 1100, 'T': 1089, 'max_update_p': 0.139, 'min_update_p': 0, 's': 9.130000000000008, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'gamma': 0.957, 'epsilon_init': 0.9, 'epsilon_decay': 0.002, 'buffer_size': 1000, 'batch_size': 64, 'epochs': 7, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': 20, 'number_of_state_bits_ta': 5, 'update_grad': 0.173, 'update_freq': -1, 'dataset_file_name': 'observation_data'}

env = gym.make("Acrobot-v1")
#env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy


save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

agent.online_policy.tms[0].set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.online_policy.tms[1].set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])
agent.online_policy.tms[2].set_params(tms[2]['ta_state'], tms[2]['clause_sign'], tms[2]['clause_output'], tms[2]['feedback_to_clauses'])

test_policy(save_file, agent.online_policy, config['env_name'])
