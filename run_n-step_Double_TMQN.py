import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_Double_TMQN import TMQN
from algorithms.policy.RTM import Policy

config = {'algorithm': 'n_step_Double_TMQN', 'soft_update_type': 'soft_update_2', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 830, 's': 7, 'y_max': 60, 'y_min': 40, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 15, 'gamma': 0.99, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'number_of_state_bits_ta': 10, 'update_grad': 0.05, 'update_freq': 6}

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

save_file = f'results/n_step_Double_TMQN/{agent.run_id}/final_test_results'
tms = torch.load(f'results/n_step_Double_TMQN/{agent.run_id}/best')

agent.target_policy.tm1.set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.target_policy.tm2.set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[1]['clause_output'], tms[1]['feedback_to_clauses'])

test_policy(save_file, agent.target_policy)
