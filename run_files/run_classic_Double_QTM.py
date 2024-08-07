import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Networks.QTM_classic import DoubleQTM as QTM
from algorithms.policy.RTM import Policy

config = {'env_name': 'Cartpole', 'algorithm': 'QTM', 'nr_of_clauses': 960, 'T': 451, 'max_update_p': 0.5, 'soft_update_type': 'soft_update_a', 'clause_update_p': 1.00,
          'min_update_p': 0, 's': 5.75, 'y_max': 100, 'y_min': 20, 'device': 'CPU', 'bits_per_feature': 5, "n_steps": 20,
          'gamma': 0.991, 'epsilon_init': 0.8, 'epsilon_decay': 0.009, "epsilon_min": 0, 'buffer_size': 10_000,
          'threshold': 100,
          "sampling_iterations": 3, 'sample_size': 64, 'test_freq': 25, 'save': True, 'seed': 42,
          'number_of_state_bits_ta': 3, 'dataset_file_name': 'cartpole_obs_data'}

print(config)

env = gym.make("CartPole-v1")
# env = gym.make("Acrobot-v1")

agent = QTM(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.online_policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'],
                                          tms[i]['feedback_to_clauses'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}'

test_policy(save_file, agent.online_policy, config["env_name"])
