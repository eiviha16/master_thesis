import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Tsetlin_Actor_Critic.TAC import TAC
from algorithms.policy.CTM import ActorCriticPolicy as Policy


config = {"env_name": "Cartpole", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_b', 'epsilon_init': 1,
          'epsilon_decay': 0.001, "epsilon_min": 0, "train_freq": 100, 'update_freq': 20, 'gamma': 0.99, 'buffer_size': 10000,
          'actor': {'nr_of_clauses': 2000, 'T': 1800, 's': 3.68, 'device': 'CPU', 'bits_per_feature': 4, 'seed': 42,
                    'number_of_state_bits_ta': 4},
          'critic': {'max_update_p': 1.0, 'nr_of_clauses': 2050, 'T': 1413, 's': 3.47, 'y_max': 130,
                     'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 4, 'seed': 42,
                     'number_of_state_bits_ta': 4}, 'batch_size': 1, 'sample_size': 64, 'test_freq': 1,
          'save': True, 'threshold': -495, 'dataset_file_name': 'cartpole_obs_data'}

print(config)

env = gym.make("CartPole-v1")

agent = TAC(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy

tm = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
test_policy(save_file, agent.policy.actor, config['env_name'])
