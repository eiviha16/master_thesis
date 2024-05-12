import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Tsetlin_Actor_Critic.TAC import TAC
from algorithms.policy.CTM import ActorCriticPolicy as Policy


config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_b', 'epsilon_init': 1.0, 'epsilon_decay': 0.001, 'update_freq': 4, 'gamma': 0.982, 'buffer_size': 10000, 'actor': {'nr_of_clauses': 1260, 'T': 1018, 's': 2.99, 'device': 'CPU', 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.05, 'nr_of_clauses': 1450, 'T': 1039, 's': 3.370000000000006, 'y_max': 100, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, 'seed': 42, 'number_of_state_bits_ta': 5}, 'batch_size': 1, 'sampling_iterations': 64, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
print(config)

env = gym.make("Acrobot-v1")

agent = TAC(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

tm = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
test_policy(save_file, agent.policy.actor, config['env_name'])

