import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_Double_QTM import TMQN
from algorithms.policy.RTM import Policy

config = {"env_name": "acrobot", 'algorithm': 'n_step_Double_QTM_b', 'soft_update_type': 'soft_update_2', 'n_steps': 14, 'nr_of_clauses': 1860, 'T': 613, 'max_update_p': 0.151, 'min_update_p': 0, 's': 3.6000000000000023, 'y_max': -10, 'y_min': -70, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'gamma': 0.93, 'epsilon_init': 0.7, 'epsilon_decay': 0.008, 'buffer_size': 4000, 'batch_size': 48, 'epochs': 1, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': -495, 'number_of_state_bits_ta': 6, 'update_grad': -1, 'update_freq': 3, 'dataset_file_name': 'acrobot_obs_data'}

#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")

agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.online_policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.online_policy, config['env_name'])
