import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.Double_QTM import TMQN
from algorithms.policy.RTM import Policy

config = {'env_name': "cartpole", 'algorithm': 'Double_QTM_a',  'soft_update_type': 'soft_update_1', 'nr_of_clauses': 1180, 'T': 1050, 'max_update_p': 0.178, 'min_update_p': 0, 's': 6.210000000000004, 'y_max': 75, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'gamma': 0.976, 'epsilon_init': 0.9, 'epsilon_decay': 0.007, 'buffer_size': 6500, 'batch_size': 96, 'epochs': 5, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': 20, 'number_of_state_bits_ta': 6, 'update_grad': 0.618, 'update_freq': -1, 'dataset_file_name': 'observation_data'}
print(config)

#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = TMQN(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy


tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.target_policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}'

test_policy(save_file, agent.target_policy, config["env_name"])
