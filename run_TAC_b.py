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
actor = {'nr_of_clauses': 840, 'T': int(840 * 0.03), 's': 2.58, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 9, "seed": 42, 'number_of_state_bits_ta': 5}
#actor = {'nr_of_classes': 2, 'nr_of_clauses': 1060, 'T': int(1060 * 0.2), 's': 2.54, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 9}
#critic = {'nr_of_clauses': 1150, 'T': int(1150 * 0.54), 's': 6.34, 'y_max': 65, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 8}
critic = {'max_update_p': 0.153, 'nr_of_clauses': 1650, 'T': int(1650 * 0.68), 's': 7.33, 'y_max': 60, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 6, "seed": 42, 'number_of_state_bits_ta': 7}
#critic = {'nr_of_clauses': 1900, 'T': int(1900 * 0.19), 's': 5.91, 'y_max': 65, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, "seed": 42, 'number_of_state_bits_ta': 8}
#config = {'algorithm': 'TM_DDPG_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.906, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7500, 'exploration_prob_init': 0.3, 'exploration_prob_decay': 0.004, 'soft_update_type': 'soft_update_2', 'gamma': 0.948, 'update_grad': -1, 'update_freq': 5, 'actor': actor, 'critic': critic, 'batch_size': 96, 'epochs': 3, 'test_freq': 1, "save": True, "dataset_file_name": "observation_data"}
#config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7092, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'gamma': 0.913, 'update_grad': -1, 'update_freq': 7, 'actor': actor, 'critic': critic, 'batch_size': 16, 'epochs': 2, 'test_freq': 1, "save": True}
#run 5 without initialization
#run 6 with initialization
print(config)

env = gym.make("CartPole-v1")


agent = DDPG(env, Policy, config)
agent.learn(nr_of_episodes=680)

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