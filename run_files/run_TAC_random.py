


import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Tsetlin_actor_critic.TAC_random import TAC
from algorithms.policy.CTM import ActorCriticPolicy as Policy

config = {'env_name': "acrobot", 'algorithm': 'TAC_random', 'soft_update_type': 'soft_update_a', 'epsilon_init': 0.7, 'epsilon_decay': 0.005, 'update_grad': 0.8, 'gamma': 0.937, 'buffer_size': 5000, 'actor': {'nr_of_clauses': 1780, 'T': 1388, 's': 3.5100000000000025, 'device': 'CPU', 'bits_per_feature': 5, 'seed': 42, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.018000000000000002, 'nr_of_clauses': 1850, 'T': 499, 's': 4.5000000000000036, 'y_max': -5, 'y_min': -65, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 112, 'sampling_iterations': 2, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}

print(config)
env = gym.make("Acrobot-v1")
#env = gym.make("CartPole-v1")

agent = TAC(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

tm = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor, config["env_name"])
