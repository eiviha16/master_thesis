import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

"""actor = {"max_update_p": 0.005, "min_update_p": 0.0004, 'nr_of_clauses': 1120, 'T': int(1120 * 0.74), 's': 2.35, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 10, 'number_of_state_bits_ta': 3}
critic = {"max_update_p": 0.059, "min_update_p": 0.0, 'nr_of_clauses': 930, 'T': int(930 * 0.36), 's': 3.89, 'y_max': 31.5, 'y_min': 0.3,  'bits_per_feature': 8, 'number_of_state_bits_ta': 6}
config = {'algorithm': 'TM_PPO', "n_timesteps": 430, 'gamma': 0.959, 'lam': 0.979, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 64, 'epochs': 4, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data"} #"dataset_file_name": "acrobot_obs_data"}
"""#change gamma and lambda
"""actor = {"max_update_p": 0.024, "min_update_p": 0.0009, 'nr_of_clauses': 1110, 'T': int(1110 * 0.36), 's': 2.42, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 5, 'number_of_state_bits_ta': 5}
critic = {"max_update_p": 0.043, "min_update_p": 0.0, 'nr_of_clauses': 1030, 'T': int(1030 * 0.56), 's': 2.03, 'y_max': 14, 'y_min': 0.5,  'bits_per_feature': 5, 'number_of_state_bits_ta': 4}
config = {'algorithm': 'TM_PPO', "n_timesteps": 430, 'gamma': 0.944, 'lam': 0.966, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 384, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data"} #"dataset_file_name": "acrobot_obs_data"}
"""

"""actor = {"max_update_p": 0.068, "min_update_p": 0.0009, 'nr_of_clauses': 1140, 'T': int(1140 * 0.5), 's': 1.92, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 5, 'number_of_state_bits_ta': 7}
critic = {"max_update_p": 0.056, "min_update_p": 0.0, 'nr_of_clauses': 1120, 'T': int(1120 * 0.57), 's': 3.28, 'y_max': 25.5, 'y_min': 0.8,  'bits_per_feature': 8, 'number_of_state_bits_ta': 4}
config = {'env_name': "cartpole", 'algorithm': 'TPPO', "n_timesteps": 430, 'gamma': 0.945, 'lam': 0.968, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 64, 'epochs': 2, 'test_freq': 1, "save": True, "seed": 42, "dataset_file_name": "observation_data", "threshold": -1000} #"dataset_file_name": "acrobot_obs_data"}
"""
"""actor = {"max_update_p": 0.092, "min_update_p": 0.0008, 'nr_of_clauses': 1630, 'T': int(1630 * 0.33), 's': 1.89, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 11, 'number_of_state_bits_ta': 4}
critic = {"max_update_p": 0.016, "min_update_p": 0.0, 'nr_of_clauses': 1740, 'T': int(1740 * 0.6), 's': 1.53, 'y_max': -0.7, 'y_min': -17.7,  'bits_per_feature': 8, 'number_of_state_bits_ta': 4}
config = {'env_name': "acrobot", 'algorithm': 'TPPO', "n_timesteps": 16, 'gamma': 0.98, 'lam': 0.977, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 64, 'epochs': 4, 'test_freq': 1, "save": False, "seed": 42, "dataset_file_name": "acrobot_obs_data", "threshold": -1000} #"dataset_file_name": "acrobot_obs_data"}"observation_data"
"""
actor = {"max_update_p": 0.016, "min_update_p": 0.0008, 'nr_of_clauses': 1850, 'T': int(1850 * 0.34), 's': 1.76, 'y_max': 100, 'y_min': 0,  'bits_per_feature': 6, 'number_of_state_bits_ta': 6}
critic = {"max_update_p": 0.075, "min_update_p": 0.0, 'nr_of_clauses': 1340, 'T': int(1340 * 0.78), 's': 1.65, 'y_max': -1.0, 'y_min': -29.1,  'bits_per_feature': 10, 'number_of_state_bits_ta': 7}
config = {'env_name': "acrobot", 'algorithm': 'TPPO', "n_timesteps": 12, 'gamma': 0.951, 'lam': 0.944, "actor": actor, "critic": critic, 'device': 'CPU', 'weighted_clauses': False,  'batch_size': 128, 'epochs': 2, 'test_freq': 5, "save": True, "seed": 42, "dataset_file_name": "acrobot_obs_data", "threshold": -1000} #"dataset_file_name": "acrobot_obs_data"}"observation_data"


#run_908 has been initialized! 500.0 (best)
print(config)
#run_895 - 500.0
env = gym.make("Acrobot-v1")
#env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=2500)

from test_policy import test_policy

#agent.policy.actor.tms[0].set_state()
#agent.policy.actor.tms[1].set_state()
#save_file = f'../results/TM_PPO/{agent.run_id}'
save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'
#save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/run_2/final_test_results'

#tms = torch.load(f'../results/TM_PPO/{agent.run_id}/best')
tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')
#tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/run_2/best')

for i in range(len(tms)):
    #eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy.actor, config['env_name'])