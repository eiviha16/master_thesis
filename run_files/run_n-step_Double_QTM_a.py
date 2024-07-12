import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Networks.QTM import DoubleQTM as QTM
from algorithms.policy.RTM import Policy

config = {"env_name": "cartpole", 'algorithm': 'n_step_Double_QTM_a', 'soft_update_type': 'soft_update_a',
          'n_steps': 20, 'nr_of_clauses': 1100, 'T': 1089, 'max_update_p': 0.139, 'min_update_p': 0,
          's': 3.130000000000008, 'y_max': 110, 'y_min': 20, 'device': 'CPU', 'bits_per_feature': 6, 'gamma': 0.99,
          'epsilon_init': 0.9, 'epsilon_decay': 0.002,  "epsilon_min": 0, "train_freq": 10, 'buffer_size': 10000, 'batch_size': 1, 'sample_size': 64,
          'test_freq': 10, 'save': True, 'seed': 42, 'threshold': 20, 'number_of_state_bits_ta': 5,
          'clause_update_p': 0.173, 'dataset_file_name': 'acrobot_obs_data'}
print(config)

#env = gym.make("Acrobot-v1")
env = gym.make("CartPole-v1")

agent = QTM(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
for i in range(len(tms)):
    agent.online_policy.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'],
                                          tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.online_policy, config['env_name'])
