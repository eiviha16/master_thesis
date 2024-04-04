import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

actor = {"max_update_p": 0.047, "min_update_p": 0.0004, 'nr_of_clauses': 1430, 'T': int(1430 * 0.36), 's': 2.72, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 8, 'number_of_state_bits_ta': 4}
critic = {"max_update_p": 0.012, "min_update_p": 0.0, 'nr_of_clauses': 1680, 'T': int(1680 * 0.77), 's': 3.49, 'y_max': -0.6, 'y_min': -24.5,  'bits_per_feature': 8, 'number_of_state_bits_ta': 6}
config = {"env_name": "acrobot", 'algorithm': 'TPPO', "n_timesteps": 20, 'gamma': 0.956, 'lam': 0.976, "actor": actor, "critic": critic, "threshold": -500, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 144, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "acrobot_obs_data"}

"""actor = {"max_update_p": 0.013, "min_update_p": 0.0004, 'nr_of_clauses': 1490, 'T': int(1490 * 0.72), 's': 1.67, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 6, 'number_of_state_bits_ta': 3}
critic = {"max_update_p": 0.029, "min_update_p": 0.0, 'nr_of_clauses': 1240, 'T': int(1240 * 0.64), 's': 2.24, 'y_max': -1, 'y_min': -30.5,  'bits_per_feature': 7, 'number_of_state_bits_ta': 6}
config = {'algorithm': 'TM_PPO', "n_timesteps": 24, 'gamma': 0.969, 'lam': 0.946, "actor": actor, "critic": critic, "threshold": -500, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 144, 'epochs': 3, 'test_freq': 1, "save": False, "seed": 42, "dataset_file_name": "acrobot_obs_data"}
"""
"""actor = {"max_update_p": 0.016, "min_update_p": 0.0003, 'nr_of_clauses': 960, 'T': int(960 * 0.3), 's': 2.58, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 14, 'number_of_state_bits_ta': 4}
critic = {"max_update_p": 0.026, "min_update_p": 0.0, 'nr_of_clauses': 980, 'T': int(980 * 0.44), 's': 3.74, 'y_max': 34, 'y_min': 0.4,  'bits_per_feature': 11, 'number_of_state_bits_ta': 5}
config = {'algorithm': 'TM_PPO', "n_timesteps": 430, 'gamma': 0.967, 'lam': 0.947, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 224, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data"} #"dataset_file_name": "acrobot_obs_data"}
"""#change gamma and lambda
"""actor = {"max_update_p": 0.024, "min_update_p": 0.0009, 'nr_of_clauses': 1110, 'T': int(1110 * 0.36), 's': 2.42, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 5, 'number_of_state_bits_ta': 5}
critic = {"max_update_p": 0.043, "min_update_p": 0.0, 'nr_of_clauses': 1030, 'T': int(1030 * 0.56), 's': 2.03, 'y_max': 14, 'y_min': 0.5,  'bits_per_feature': 5, 'number_of_state_bits_ta': 4}
config = {'algorithm': 'TM_PPO', "n_timesteps": 430, 'gamma': 0.944, 'lam': 0.966, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 384, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data"} #"dataset_file_name": "acrobot_obs_data"}
"""
config = {"env_name": "cartpole", 'algorithm': 'TPPO', 'gamma': 0.952, 'lam': 0.995, 'device': 'CPU', 'weighted_clauses': False, 'actor': {'max_update_p': 0.025, 'min_update_p': 0.0007000000000000001, 'nr_of_clauses': 1000, 'T': 830, 's': 1.2000000000000002, 'y_max': 100, 'y_min': 0, 'bits_per_feature': 14, 'number_of_state_bits_ta': 4}, 'critic': {'max_update_p': 0.093, 'min_update_p': 0.0, 'nr_of_clauses': 950, 'T': 361, 's': 2.200000000000001, 'y_max': 14.5, 'y_min': 0.6000000000000001, 'bits_per_feature': 11, 'number_of_state_bits_ta': 5}, 'batch_size': 320, 'epochs': 2, 'test_freq': 1, 'save': True, 'seed': 42, 'threshold': 20, 'n_timesteps': 8, 'dataset_file_name': 'observation_data'}

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

test_policy(save_file, agent.policy.actor, config['env_name'])

"""CARTPOLE - 1000 episodes per sweep
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "gamma": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "lam": {"values": list(np.arange(0.94, 0.98, 0.001))},
        "n_timesteps": {"values": list(range(8, 32, 4))},
        "batch_size": {"values": list(range(16, 512, 16))},
        "epochs": {"values": list(range(1, 5, 1))},

        "a_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "a_nr_of_clauses": {"values": list(range(900, 1200, 10))},
        "a_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "a_bits_per_feature": {"values": list(range(5, 15, 1))},
        "a_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "a_max_update_p": {"values": list(np.arange(0.001, 0.2000, 0.001))},
        "a_min_update_p": {"values": list(np.arange(0.0001, 0.001, 0.0001))},

        "c_t": {"values": list(np.arange(0.3, 0.9, 0.01))},
        "c_nr_of_clauses": {"values": list(range(900, 1200, 10))},
        "c_specificity": {"values": list(np.arange(1.0, 4.0, 0.01))},
        "c_bits_per_feature": {"values": list(range(5, 15, 1))},
        "c_number_of_state_bits_ta": {"values": list(range(3, 8, 1))},
        "c_max_update_p": {"values": list(np.arange(0.001, 0.200, 0.001))},
        "c_y_max": {"values": list(np.arange(5.5, 35, 0.5))},
        "c_y_min": {"values": list(np.arange(0.0, 1.0, 0.1))},
    }
}

"""