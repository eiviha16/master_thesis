


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
config = {"env_name": "cartpole", 'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.005, 'update_grad': 0.722, 'gamma': 0.976, 'buffer_size': 7000, 'actor': {'nr_of_clauses': 820, 'T': 418, 's': 1.06, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 9}, 'critic': {'max_update_p': 0.018000000000000002, 'nr_of_clauses': 1050, 'T': 388, 's': 2.4700000000000015, 'y_max': 75, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 3}, 'batch_size': 80, 'epochs': 6, 'test_freq': 1, 'save': True, 'threshold': 20, 'dataset_file_name': 'observation_data'}
print(config)

#env = gym.make("Acrobot-v1")
env = gym.make("CartPole-v1")

############## MIGHT BE WORTH RUNNING ######################

#config = {'algorithm': 'TAC_a', 'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.5, 'exploration_prob_decay': 0.005, 'update_grad': 0.062, 'gamma': 0.904, 'buffer_size': 7000, 'actor': {'nr_of_clauses': 1380, 'T': 1297, 's': 8.870000000000008, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.074, 'nr_of_clauses': 1100, 'T': 121, 's': 9.280000000000008, 'y_max': -10, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 3}, 'batch_size': 80, 'epochs': 3, 'test_freq': 1, 'save': False, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {'algorithm': 'TAC_a',  'soft_update_type': 'soft_update_1', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.009, 'update_grad': 0.583, 'gamma': 0.919, 'buffer_size': 5000, 'actor': {'nr_of_clauses': 1820, 'T': 1365, 's': 1.48, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'seed': 42, 'number_of_state_bits_ta': 3}, 'critic': {'max_update_p': 0.005, 'nr_of_clauses': 1200, 'T': 744, 's': 4.15, 'y_max': -10, 'y_min': -60, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, 'seed': 42, 'number_of_state_bits_ta': 4}, 'batch_size': 112, 'epochs': 3, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
###########

agent = DDPG(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy

tm = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor)













exit(0)
#actor = {'nr_of_classes': 2, 'nr_of_clauses': 1160, 'T': int(1160 * 0.52), 's': 4.5, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 9}
actor = {'nr_of_clauses': 940, 'T': int(940 * 0.44), 's': 1.85, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, "seed": 42, 'number_of_state_bits_ta': 6}
#actor = {'nr_of_classes': 2, 'nr_of_clauses': 1060, 'T': int(1060 * 0.2), 's': 2.54, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 9}
#critic = {'nr_of_clauses': 1150, 'T': int(1150 * 0.54), 's': 6.34, 'y_max': 65, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 8}
critic = {'max_update_p': 0.127, 'nr_of_clauses': 1450, 'T': int(1450 * 0.06), 's': 5.36, 'y_max': 60, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 9}
#critic = {'nr_of_clauses': 1900, 'T': int(1900 * 0.19), 's': 5.91, 'y_max': 65, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, "seed": 42, 'number_of_state_bits_ta': 8}
#config = {'algorithm': 'TM_DDPG_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.906, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
config = {'algorithm': 'TAC_a', 'buffer_size': 5000, 'exploration_prob_init': 0.5, 'exploration_prob_decay': 0.008, 'soft_update_type': 'soft_update_1', 'gamma': 0.906, 'update_grad': 0.981, 'actor': actor, 'critic': critic, 'batch_size': 80, 'epochs': 6, 'test_freq': 1, "save": False, "dataset_file_name": "observation_data"}
#config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7092, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'gamma': 0.913, 'update_grad': -1, 'update_freq': 7, 'actor': actor, 'critic': critic, 'batch_size': 16, 'epochs': 2, 'test_freq': 1, "save": True}
#run 5 without initialization
#run 6 with initialization
print(config)
#run 1152 achieves 500.0
#current: run 1154 (500.0)
env = gym.make("CartPole-v1")


agent = DDPG(env, Policy, config)
agent.learn(nr_of_episodes=5_000)

from test_policy import test_policy

tm = torch.load(f'results/TM_DDPG_2/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'results/TM_DDPG_2/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor)

"""CARTPOLE 1000 episodes per sweep
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_grad": {"values": list(np.arange(0.001, 1.0, 0.001))},
        "batch_size": {"values": list(range(16, 128, 16))},
        "epochs": {"values": list(range(1, 8, 1))},

        "a_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "a_nr_of_clauses": {"values": list(range(800, 1200, 20))},
        "a_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},

        "c_t": {"values": list(np.arange(0.01, 1.00, 0.01))},
        "c_nr_of_clauses": {"values": list(range(800, 2000, 50))},
        "c_specificity": {"values": list(np.arange(1.0, 10.00, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 10, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.2, 0.001))},

        "c_y_max": {"values": list(range(60, 80, 5))},
        "c_y_min": {"values": list(range(20, 40, 5))},
        "buffer_size": {"values": list(range(500, 10_000, 500))},
        "exploration_p_decay": {"values": list(np.arange(0.001, 0.01, 0.001))},
        "exploration_p_init": {"values": list(np.arange(0.2, 1.00, 0.1))},
    }
}

"""