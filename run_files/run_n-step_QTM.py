import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Networks.QTM import SingleQTM as QTM
from algorithms.policy.RTM import Policy

config = {"env_name": "cartpole", 'algorithm': 'n_step_QTM', 'n_steps': 30, 'nr_of_clauses': 1200, 'T': 1058,
          'max_update_p': 0.106, 'min_update_p': 0, 's': 2.270000000000002, 'y_max': 100, 'y_min': 20, 'device': 'CPU',
          'bits_per_feature': 5, 'gamma': 0.983, 'epsilon_init': 0.9, 'epsilon_decay': 0.001, "epsilon_min": 0.01, "train_freq": 100,
          'buffer_size': 100_000, 'threshold': -200, 'sample_size': 64, 'test_freq': 5,
          'save': True, 'seed': 42, 'number_of_state_bits_ta': 5, 'dataset_file_name': 'cartpole_obs_data'}
print(config)

env = gym.make("CartPole-v1")
#env = gym.make("Acrobot-v1")

agent = QTM(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'],
                                   tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy, config['env_name'])
