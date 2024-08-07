import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Networks.QTM import DoubleQTM as QTM
from algorithms.policy.RTM import Policy

config = {"env_name": "acrobot", 'algorithm': 'Double_QTM_b', 'soft_update_type': 'soft_update_b',
          'nr_of_clauses': 1580, 'T': 711, 'max_update_p': 0.055, 'min_update_p': 0, 's': 3.8000000000000025,
          'y_max': -5, 'y_min': -55, 'device': 'CPU', 'bits_per_feature': 8, 'gamma': 0.959, 'epsilon_init': 0.7, "n_steps": -1,
          'epsilon_decay': 0.004, "epsilon_min": 0, "train_freq": 100, 'buffer_size': 9500, 'batch_size': 1, 'sample_size': 64, 'test_freq': 1,
          'save': True, 'seed': 42, 'threshold': -495, 'number_of_state_bits_ta': 7, 'update_freq': 5,
          'dataset_file_name': 'acrobot_obs_data'}
print(config)

env = gym.make("CartPole-v1")
#env = gym.make("Acrobot-v1")

agent = QTM(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.online_policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'],
                                          tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.online_policy, config['env_name'])
