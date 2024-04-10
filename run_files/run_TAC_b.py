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
#actor = {'nr_of_clauses': 1660, 'T': int(1660 * 0.25), 's': 7.13, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 3}
#actor = {'nr_of_clauses': 1780, 'T': int(1780 * 0.99), 's': 9.89, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, "seed": 42, 'number_of_state_bits_ta': 4}
#actor = {'nr_of_classes': 2, 'nr_of_clauses': 1060, 'T': int(1060 * 0.2), 's': 2.54, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 9}
#critic = {'nr_of_clauses': 1150, 'T': int(1150 * 0.54), 's': 6.34, 'y_max': 65, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 8}
#critic = {'max_update_p': 0.153, 'nr_of_clauses': 1650, 'T': int(1650 * 0.68), 's': 7.33, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, "seed": 42, 'number_of_state_bits_ta': 7}
#critic = {'max_update_p': 0.012, 'nr_of_clauses': 1650, 'T': int(1650 * 0.96), 's': 1.01, 'y_max': -25, 'y_min': -75, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 9}
#critic = {'max_update_p': 0.082, 'nr_of_clauses': 1500, 'T': int(1450 * 0.48), 's': 2.18, 'y_max': -30, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, "seed": 42, 'number_of_state_bits_ta': 4}
#critic = {'nr_of_clauses': 1900, 'T': int(1900 * 0.19), 's': 5.91, 'y_max': 65, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, "seed": 42, 'number_of_state_bits_ta': 8}
#config = {'algorithm': 'TM_DDPG_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.906, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
#config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7500, 'exploration_prob_init': 0.3, 'exploration_prob_decay': 0.004, 'soft_update_type': 'soft_update_2', 'gamma': 0.948, 'update_grad': -1, 'update_freq': 5, 'actor': actor, 'critic': critic, 'batch_size': 96, 'epochs': 3, 'test_freq': 1, "save": False, "threshold": 0, "dataset_file_name": "observation_data"}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'buffer_size': 8500, 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.006, 'soft_update_type': 'soft_update_2', 'gamma': 0.949, 'update_freq': 8, 'actor': actor, 'critic': critic, 'batch_size': 112, 'epochs': 1, 'test_freq': 1, "save": True, "threshold": -500, "dataset_file_name": "acrobot_obs_data"}#observation_data"}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'buffer_size': 7500, 'exploration_prob_init': 0.6, 'exploration_prob_decay': 0.006, 'soft_update_type': 'soft_update_2', 'gamma': 0.962, 'update_freq': 3, 'actor': actor, 'critic': critic, 'batch_size': 96, 'epochs': 2, 'test_freq': 1, "save": True, "threshold": -500, "dataset_file_name": "acrobot_obs_data"}#observation_data"}
#config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7092, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'gamma': 0.913, 'update_grad': -1, 'update_freq': 7, 'actor': actor, 'critic': critic, 'batch_size': 16, 'epochs': 2, 'test_freq': 1, "save": True}
#run 5 without initialization
#run 6 with initialization

#config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.6, 'exploration_prob_decay': 0.003, 'update_freq': 8, 'gamma': 0.955, 'buffer_size': 9000, 'actor': {'nr_of_clauses': 1160, 'T': 707, 's': 1.6600000000000006, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 10, 'seed': 42, 'number_of_state_bits_ta': 5}, 'critic': {'max_update_p': 0.059, 'nr_of_clauses': 1600, 'T': 192, 's': 6.860000000000005, 'y_max': -15, 'y_min': -65, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 80, 'epochs': 4, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.7, 'exploration_prob_decay': 0.001, 'update_freq': 4, 'gamma': 0.982, 'buffer_size': 5500, 'actor': {'nr_of_clauses': 1860, 'T': 818, 's': 2.9800000000000018, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 3}, 'critic': {'max_update_p': 0.07, 'nr_of_clauses': 1650, 'T': 1039, 's': 8.370000000000006, 'y_max': -10, 'y_min': -80, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 5}, 'batch_size': 16, 'epochs': 3, 'test_freq': 5, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_b',  'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.5, 'exploration_prob_decay': 0.008, 'update_freq': 6, 'gamma': 0.921, 'buffer_size': 6500, 'actor': {'nr_of_clauses': 1160, 'T': 962, 's': 1.09, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'seed': 42, 'number_of_state_bits_ta': 7}, 'critic': {'max_update_p': 0.052000000000000005, 'nr_of_clauses': 1300, 'T': 156, 's': 8.480000000000008, 'y_max': -25, 'y_min': -50, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'seed': 42, 'number_of_state_bits_ta': 4}, 'batch_size': 96, 'epochs': 7, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.7999999999999999, 'exploration_prob_decay': 0.007, 'update_freq': 6, 'gamma': 0.969, 'buffer_size': 2500, 'actor': {'nr_of_clauses': 1300, 'T': 1105, 's': 1.3000000000000005, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'seed': 42, 'number_of_state_bits_ta': 7}, 'critic': {'max_update_p': 0.053000000000000005, 'nr_of_clauses': 1600, 'T': 64, 's': 2.410000000000001, 'y_max': -5, 'y_min': -55, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'seed': 42, 'number_of_state_bits_ta': 6}, 'batch_size': 80, 'epochs': 4, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {"env_name": "cartpole",'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.001, 'update_freq': 1, 'gamma': 0.96, 'buffer_size': 3000, 'actor': {'nr_of_clauses': 1020, 'T': 989, 's': 6.870000000000005, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'seed': 42, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.069, 'nr_of_clauses': 1300, 'T': 871, 's': 9.440000000000008, 'y_max': 75, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 8}, 'batch_size': 112, 'epochs': 7, 'test_freq': 1, 'threshold': 20, 'save': True, 'dataset_file_name': 'observation_data'}
#config = {"env_name": "cartpole", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.005, 'update_freq': 4, 'gamma': 0.956, 'buffer_size': 1000, 'actor': {'nr_of_clauses': 1100, 'T': 187, 's': 3.9200000000000026, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14, 'seed': 42, 'number_of_state_bits_ta': 9}, 'critic': {'max_update_p': 0.083, 'nr_of_clauses': 900, 'T': 405, 's': 3.1700000000000017, 'y_max': 75, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'seed': 42, 'number_of_state_bits_ta': 8}, 'batch_size': 112, 'epochs': 7, 'test_freq': 1, 'threshold': 20, 'save': True, 'dataset_file_name': 'observation_data'}
#config = {"env_name": "cartpole", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.8, 'exploration_prob_decay': 0.002, 'update_freq': 8, 'gamma': 0.994, 'buffer_size': 3500, 'actor': {'nr_of_clauses': 1120, 'T': 190, 's': 2.170000000000001, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'seed': 42, 'number_of_state_bits_ta': 7}, 'critic': {'max_update_p': 0.081, 'nr_of_clauses': 1400, 'T': 1078, 's': 4.120000000000003, 'y_max': 70, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 64, 'epochs': 7, 'test_freq': 1, 'threshold': 20, 'save': True, 'dataset_file_name': 'observation_data'}
#config = {"env_name": "cartpole", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.002, 'update_freq': 1, 'gamma': 0.988, 'buffer_size': 2500, 'actor': {'nr_of_clauses': 1180, 'T': 365, 's': 2.520000000000002, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, 'seed': 42, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.093, 'nr_of_clauses': 800, 'T': 496, 's': 3.950000000000003, 'y_max': 70, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 3}, 'batch_size': 64, 'epochs': 7, 'test_freq': 1, 'threshold': 20, 'save': True, 'dataset_file_name': 'observation_data'}
#config = {"env_name": "cartpole", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.9, 'exploration_prob_decay': 0.002, 'update_freq': 3, 'gamma': 0.982, 'buffer_size': 1000, 'actor': {'nr_of_clauses': 1100, 'T': 506, 's': 3.6800000000000024, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 13, 'seed': 42, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.056, 'nr_of_clauses': 1350, 'T': 513, 's': 7.470000000000006, 'y_max': 75, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'seed': 42, 'number_of_state_bits_ta': 9}, 'batch_size': 48, 'epochs': 6, 'test_freq': 1, 'threshold': 20, 'save': True, 'dataset_file_name': 'observation_data'}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.7999999999999999, 'exploration_prob_decay': 0.006, 'update_freq': 8, 'gamma': 0.94, 'buffer_size': 2500, 'actor': {'nr_of_clauses': 1300, 'T': 1274, 's': 8.680000000000007, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 11, 'seed': 42, 'number_of_state_bits_ta': 8}, 'critic': {'max_update_p': 0.033, 'nr_of_clauses': 1900, 'T': 152, 's': 1.5700000000000005, 'y_max': -5, 'y_min': -65, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, 'seed': 42, 'number_of_state_bits_ta': 3}, 'batch_size': 32, 'epochs': 4, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#config = {"env_name": "acrobot", 'algorithm': 'TAC_b', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.7999999999999999, 'exploration_prob_decay': 0.007, 'update_freq': 1, 'gamma': 0.933, 'buffer_size': 5000, 'actor': {'nr_of_clauses': 1900, 'T': 1558, 's': 5.630000000000004, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'seed': 42, 'number_of_state_bits_ta': 7}, 'critic': {'max_update_p': 0.060000000000000005, 'nr_of_clauses': 1300, 'T': 884, 's': 2.830000000000002, 'y_max': -10, 'y_min': -75, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, 'seed': 42, 'number_of_state_bits_ta': 5}, 'batch_size': 112, 'epochs': 2, 'test_freq': 1, 'save': True, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
print(config)
#untested acrobot {'algorithm': 'TM_DDPG_2', 'soft_update_type': 'soft_update_2', 'exploration_prob_init': 0.8999999999999999, 'exploration_prob_decay': 0.002, 'update_freq': 6, 'gamma': 0.924, 'buffer_size': 5000, 'actor': {'nr_of_clauses': 1500, 'T': 1440, 's': 2.5900000000000016, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, 'seed': 42, 'number_of_state_bits_ta': 9}, 'critic': {'max_update_p': 0.044, 'nr_of_clauses': 1550, 'T': 1364, 's': 7.370000000000005, 'y_max': -5, 'y_min': -70, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'seed': 42, 'number_of_state_bits_ta': 4}, 'batch_size': 32, 'epochs': 1, 'test_freq': 1, 'save': False, 'threshold': -495, 'dataset_file_name': 'acrobot_obs_data'}
#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = DDPG(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

tm = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor, config['env_name'])
"""CARTPOLE 1000 episodes per sweep
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.90, 1.00, 0.001))},
        "update_freq": {"values": list(range(1, 10, 1))},
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