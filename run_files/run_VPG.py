


import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.VPG.TM_DDPG import DDPG
from algorithms.policy.CTM import ActorCriticPolicy as Policy
#actor = {'nr_of_classes': 2, 'nr_of_clauses': 1160, 'T': int(1160 * 0.52), 's': 4.5, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 9}
#actor = {'nr_of_clauses': 840, 'T': int(840 * 0.03), 's': 2.58, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, "seed": 42, 'number_of_state_bits_ta': 5}
#actor = {'nr_of_clauses': 960, 'T': int(960 * 0.35), 's': 1.41, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, "seed": 42, 'number_of_state_bits_ta': 3}
#actor = {'nr_of_classes': 2, 'nr_of_clauses': 1060, 'T': int(1060 * 0.2), 's': 2.54, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 9}
#critic = {'nr_of_clauses': 1150, 'T': int(1150 * 0.54), 's': 6.34, 'y_max': 65, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 8}
#critic = {'max_update_p': 0.153, 'nr_of_clauses': 1650, 'T': int(1650 * 0.68), 's': 7.33, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, "seed": 42, 'number_of_state_bits_ta': 7}
#critic = {'max_update_p': 0.027, 'nr_of_clauses': 1850, 'T': int(1850 * 0.55), 's': 2.59, 'y_max': 65, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 3}
#critic = {'nr_of_clauses': 1900, 'T': int(1900 * 0.19), 's': 5.91, 'y_max': 65, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, "seed": 42, 'number_of_state_bits_ta': 8}
#config = {'algorithm': 'TM_DDPG_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.906, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
#config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7500, 'exploration_prob_init': 0.3, 'exploration_prob_decay': 0.004, 'soft_update_type': 'soft_update_2', 'gamma': 0.948, 'update_grad': -1, 'update_freq': 5, 'actor': actor, 'critic': critic, 'batch_size': 96, 'epochs': 3, 'test_freq': 1, "save": False, "threshold": 0, "dataset_file_name": "observation_data"}
#config = {"env_name": "cartpole", 'algorithm': 'TAC_a', 'buffer_size': 2500, 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.002, 'soft_update_type': 'soft_update_1', 'gamma': 0.954, 'update_grad': 0.677, 'actor': actor, 'critic': critic, 'batch_size': 112, 'epochs': 4, 'test_freq': 1, "save": True, "threshold": 0, "dataset_file_name": "observation_data"}
#config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7092, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'gamma': 0.913, 'update_grad': -1, 'update_freq': 7, 'actor': actor, 'critic': critic, 'batch_size': 16, 'epochs': 2, 'test_freq': 1, "save": True}
#run 5 without initialization
#run 6 with initialization
#config = {"env_name": "acrobot", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.7999999999999999, 'exploration_prob_decay': 0.007, 'update_grad': 0.105, 'gamma': 0.948, 'buffer_size': 4000, 'actor': {'nr_of_clauses': 1580, 'T': 726, 's': 3.9300000000000024, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'seed': 42, 'number_of_state_bits_ta': 8}, 'critic': {'max_update_p': 0.028, 'nr_of_clauses': 1800, 'T': 756, 's': 8.670000000000007, 'y_max': -5, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 96, 'epochs': 4, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.7, 'exploration_prob_decay': 0.001, 'update_freq': 4, 'gamma': 0.982, 'buffer_size': 5500, 'actor': {'nr_of_clauses': 1860, 'T': 818, 's': 2.98, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 3}, 'critic': {'max_update_p': 0.07, 'nr_of_clauses': 1650, 'T': 1039, 's': 8.37, 'y_max': -10, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 5}, 'batch_size': 16, 'epochs': 3, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.8999999999999999, 'exploration_prob_decay': 0.006, 'update_grad': 0.749, 'gamma': 0.943, 'buffer_size': 9500, 'actor': {'nr_of_clauses': 1840, 'T': 1343, 's': 1.01, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 13, 'seed': 42, 'number_of_state_bits_ta': 8}, 'critic': {'max_update_p': 0.078, 'nr_of_clauses': 1850, 'T': 1406, 's': 3.290000000000002, 'y_max': -5, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 13, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 48, 'epochs': 2, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {"env_name": "cartpole",'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.005, 'update_grad': 0.846, 'gamma': 0.936, 'buffer_size': 3500, 'actor': {'nr_of_clauses': 920, 'T': 763, 's': 2.4600000000000013, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, 'seed': 42, 'number_of_state_bits_ta': 4}, 'critic': {'max_update_p': 0.051, 'nr_of_clauses': 800, 'T': 32, 's': 7.130000000000005, 'y_max': 75, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 6}, 'batch_size': 96, 'epochs': 6, 'test_freq': 1, 'save': True, 'threshold': 20, 'dataset_file_name': 'observation_data'}
#config = {"env_name": "cartpole", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.005, 'update_grad': 0.722, 'gamma': 0.976, 'buffer_size': 7000, 'actor': {'nr_of_clauses': 820, 'T': 418, 's': 1.06, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 9}, 'critic': {'max_update_p': 0.018000000000000002, 'nr_of_clauses': 1050, 'T': 388, 's': 2.4700000000000015, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 3}, 'batch_size': 80, 'epochs': 6, 'test_freq': 1, 'save': True, 'threshold': 20, 'dataset_file_name': 'observation_data'}
#onfig = {"env_name": "cartpole", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.008, 'update_grad': 0.785, 'gamma': 0.916, 'buffer_size': 3000, 'actor': {'nr_of_clauses': 980, 'T': 637, 's': 3.8600000000000025, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'seed': 42, 'number_of_state_bits_ta': 9}, 'critic': {'max_update_p': 0.069, 'nr_of_clauses': 1550, 'T': 744, 's': 5.400000000000004, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 80, 'epochs': 4, 'test_freq': 1, 'save': True, 'threshold': 20, 'dataset_file_name': 'observation_data'}
config = {"env_name": "acrobot", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.8999999999999999, 'exploration_prob_decay': 0.001, 'update_grad': 0.139, 'gamma': 0.944, 'buffer_size': 5000, 'actor': {'nr_of_clauses': 1960, 'T': 450, 's': 7.880000000000006, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'seed': 42, 'number_of_state_bits_ta': 8}, 'critic': {'max_update_p': 0.053000000000000005, 'nr_of_clauses': 1150, 'T': 954, 's': 6.700000000000005, 'y_max': -10, 'y_min': -70, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'seed': 42, 'number_of_state_bits_ta': 8}, 'batch_size': 112, 'epochs': 3, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
"""Mean reward: -87.98
Mean std: 44.62465237959844
Actions: 8897"""
#config = {"env_name": "acrobot", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.8999999999999999, 'exploration_prob_decay': 0.001, 'update_grad': 0.139, 'gamma': 0.944, 'buffer_size': 5000, 'actor': {'nr_of_clauses': 1960, 'T': 450, 's': 7.880000000000006, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'seed': 42, 'number_of_state_bits_ta': 8}, 'critic': {'max_update_p': 0.053000000000000005, 'nr_of_clauses': 1150, 'T': 954, 's': 6.700000000000005, 'y_max': -10, 'y_min': -70, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'seed': 42, 'number_of_state_bits_ta': 8}, 'batch_size': 112, 'epochs': 3, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.7, 'exploration_prob_decay': 0.001, 'update_freq': 4, 'gamma': 0.982, 'buffer_size': 5500, 'actor': {'nr_of_clauses': 1860, 'T': 818, 's': 2.9800000000000018, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 3}, 'critic': {'max_update_p': 0.07, 'nr_of_clauses': 1650, 'T': 1039, 's': 8.370000000000006, 'y_max': -10, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 5}, 'batch_size': 16, 'epochs': 3, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
"""Mean reward: -91.51
Mean std: 19.226281491749777
Actions: 9251"""
print(config)
#not tested acrobot {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.7, 'exploration_prob_decay': 0.002, 'update_grad': 0.789, 'gamma': 0.959, 'buffer_size': 1000, 'actor': {'nr_of_clauses': 1600, 'T': 704, 's': 8.590000000000007, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 4}, 'critic': {'max_update_p': 0.093, 'nr_of_clauses': 1000, 'T': 880, 's': 1.760000000000001, 'y_max': -5, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'seed': 42, 'number_of_state_bits_ta': 8}, 'batch_size': 16, 'epochs': 4, 'test_freq': 1, 'save': False, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
env = gym.make("Acrobot-v1")
#env = gym.make("CartPole-v1")

############## MIGHT BE WORTH RUNNING ######################

#config = {'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.5, 'exploration_prob_decay': 0.005, 'update_grad': 0.062, 'gamma': 0.904, 'buffer_size': 7000, 'actor': {'nr_of_clauses': 1380, 'T': 1297, 's': 8.870000000000008, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.074, 'nr_of_clauses': 1100, 'T': 121, 's': 9.280000000000008, 'y_max': -10, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 3}, 'batch_size': 80, 'epochs': 3, 'test_freq': 1, 'save': False, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {'algorithm': 'TAC_a',  'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.009, 'update_grad': 0.583, 'gamma': 0.919, 'buffer_size': 5000, 'actor': {'nr_of_clauses': 1820, 'T': 1365, 's': 1.48, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'seed': 42, 'number_of_state_bits_ta': 3}, 'critic': {'max_update_p': 0.005, 'nr_of_clauses': 1200, 'T': 744, 's': 4.15, 'y_max': -10, 'y_min': -60, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, 'seed': 42, 'number_of_state_bits_ta': 4}, 'batch_size': 112, 'epochs': 3, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
###########

agent = DDPG(env, Policy, config)
agent.learn(nr_of_episodes=1000)

from test_policy import test_policy

tm = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor, config["env_name"])





exit(0)

import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

actor = {"max_update_p": 0.035, "min_update_p": 0.0005, 'nr_of_clauses': 1380, 'T': int(1380 * 0.6), 's': 1.33, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 14, 'number_of_state_bits_ta': 4}
critic = {"max_update_p": 0.017, "min_update_p": 0.0, 'nr_of_clauses': 980, 'T': int(980 * 0.49), 's': 1.12, 'y_max': 30, 'y_min': 0.4,  'bits_per_feature': 7, 'number_of_state_bits_ta': 3}
config = {"env_name": "cartpole", 'algorithm': 'TPPO', "n_timesteps": 124, 'gamma': 0.968, 'lam': 0.974, "actor": actor, "critic": critic, "threshold": -500, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 144, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data"}

print(config)
#run_895 - 500.0
#env = gym.make("Acrobot-v1")
env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
#save_file = f'results/TM_PPO/{agent.run_id}'

#tms = torch.load(f'results/TPPO/{agent.run_id}/best')

for i in range(len(tms)):
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy.actor)

exit(0)