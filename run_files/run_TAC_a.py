


import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Tsetlin_actor_critic.TAC import TAC
from algorithms.policy.CTM import ActorCriticPolicy as Policy
config = {"env_name": "acrobot", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'epsilon_init': 0.8999999999999999, 'epsilon_decay': 0.006, 'update_grad': 0.749, 'gamma': 0.943, 'buffer_size': 9500, 'actor': {'nr_of_clauses': 1840, 'T': 1343, 's': 1.01, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 13, 'seed': 42, 'number_of_state_bits_ta': 8}, 'critic': {'max_update_p': 0.078, 'nr_of_clauses': 1850, 'T': 1406, 's': 3.290000000000002, 'y_max': -5, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 13, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 48, 'sampling_iterations': 2, 'test_freq': 5, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
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
