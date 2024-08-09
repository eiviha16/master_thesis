import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Tsetlin_Actor_Critic.TAC import TAC
from algorithms.policy.CTM import ActorCriticPolicy as Policy


#config  = {"env_name": "Cartpole", 'algorithm': 'TAC_a', 'n_steps': 98, 'soft_update_type': 'soft_update_a', 'epsilon_init': 0.9, 'epsilon_decay': 0.001, 'clause_update_p': 0.383, 'gamma': 0.99, 'buffer_size': 99500, 'actor': {'nr_of_clauses': 1550, 'T': 1348, 's': 7.970000000000006, 'device': 'CPU', 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 4}, 'critic': {'max_update_p': 0.097, 'nr_of_clauses': 1250, 'T': 537, 's': 6.440000000000005, 'y_max': 105, 'y_min': 45, 'device': 'CPU', 'bits_per_feature': 10, 'seed': 42, 'number_of_state_bits_ta': 6}, 'batch_size': 6, 'sampling_iterations': 80, 'test_freq': 1, 'save': True, 'threshold': 20, 'dataset_file_name': 'cartpole_obs_data'}
config = {"env_name": "cartpole", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_a', "n_steps": -1, 'epsilon_init': 0.9, 'epsilon_decay': 0.0001,  "epsilon_min": 0, "train_freq": 10, 'clause_update_p': 0.884, 'gamma': 0.99, 'buffer_size': 9500, 'actor': {'nr_of_clauses': 1700, 'T': 1054, 's': 6.180000000000005, 'device': 'CPU', 'bits_per_feature': 5, 'seed': 42, 'number_of_state_bits_ta': 8}, 'critic': {'max_update_p': 0.096, 'nr_of_clauses': 1900, 'T': 684, 's': 5.340000000000003, 'y_max': 100, 'y_min': 15, 'device': 'CPU', 'bits_per_feature': 5, 'seed': 42, 'number_of_state_bits_ta': 7}, 'sample_size': 128, 'test_freq': 5, 'save': True, 'threshold': 20, 'dataset_file_name': 'cartpole_obs_data'}
print(config)

#env = gym.make("Acrobot-v1")
env = gym.make("CartPole-v1")

agent = TAC(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy

tm = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor, config["env_name"])
