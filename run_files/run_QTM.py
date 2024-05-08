import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.QTM import QTM
from algorithms.policy.RTM import Policy

config = {'env_name': 'acrobot', 'algorithm': 'QTM', 'nr_of_clauses': 1240, 'T': 1165, 'max_update_p': 0.186, 'min_update_p': 0, 's': 1.3700000000000003, 'y_max': -5, 'y_min': -55, 'device': 'CPU', 'bits_per_feature': 6, 'gamma': 0.977, 'epsilon_init': 0.7, 'epsilon_decay': 0.006, 'buffer_size': 7000, 'threshold': -495, 'batch_size': 96, 'sampling_iterations': 3, 'test_freq': 5, 'save': True, 'seed': 42, 'number_of_state_bits_ta': 8, 'dataset_file_name': 'acrobot_obs_data'}

print(config)
#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = QTM(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy, config['env_name'])

