import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_Policy_Optimization.TPPO import TPPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

"""actor = {"max_update_p": 0.016, "min_update_p": 0.0008, 'nr_of_clauses': 1250, 'T': int(1250 * 0.34), 's': 1.76,
         'y_max': 100, 'y_min': 0, 'bits_per_feature': 6, 'number_of_state_bits_ta': 6}
critic = {"max_update_p": 0.075, "min_update_p": 0.0, 'nr_of_clauses': 1340, 'T': int(1240 * 0.78), 's': 1.65,
          'y_max': 100.0, 'y_min': 0, 'bits_per_feature': 4, 'number_of_state_bits_ta': 7}
config = {'env_name': "cartpole", 'algorithm': 'TPPO', "n_timesteps": 512, 'gamma': 0.99, 'lam': 0.96, "actor": actor,
          "critic": critic, 'device': 'CPU', 'epochs': 1, 'test_freq': 100, "save": True, "seed": 42,
          "dataset_file_name": "cartpole_obs_data",
          "threshold": -1000}"""  # "dataset_file_name": "acrobot_obs_data"}"observation_data"
#config = {'env_name': "cartpole", 'algorithm': 'TPPO', 'gamma': 0.982, 'lam': 0.962, 'device': 'CPU', 'actor': {'max_update_p': 0.085, 'min_update_p': 0.0008, 'nr_of_clauses': 1330, 'T': 811, 's': 7.750000000000006, 'y_max': 100, 'y_min': 0, 'bits_per_feature': 8, 'number_of_state_bits_ta': 7}, 'critic': {'max_update_p': 0.012, 'min_update_p': 0.0, 'nr_of_clauses': 1240, 'T': 545, 's': 7.130000000000005, 'y_max': 108.5, 'y_min': 0.7000000000000001, 'bits_per_feature': 4, 'number_of_state_bits_ta': 6}, 'epochs': 8, 'test_freq': 5, 'save': True, 'seed': 42, 'threshold': 0, 'n_timesteps': 156, 'dataset_file_name': 'cartpole_obs_data'}
#config = {'env_name': "cartpole", 'comment': 'newest', 'algorithm': 'TPPO', 'gamma': 0.99, 'lam': 0.961, 'device': 'CPU', 'actor': {'max_update_p': 0.015, 'min_update_p': 0.00030000000000000003, 'nr_of_clauses': 1220, 'T': 561, 's': 2.5600000000000014, 'y_max': 100, 'y_min': 0, 'bits_per_feature': 5, 'number_of_state_bits_ta': 6}, 'critic': {'max_update_p': 0.053000000000000005, 'min_update_p': 0.0, 'nr_of_clauses': 1820, 'T': 1783, 's': 2.770000000000002, 'y_max': 99.5, 'y_min': 0.30000000000000004, 'bits_per_feature': 8, 'number_of_state_bits_ta': 6}, 'epochs': 6, 'test_freq': 5, 'save': True, 'seed': 42, 'threshold': 0, 'n_timesteps': 968, 'dataset_file_name': 'cartpole_obs_data'}
config = {'env_name': "cartpole", 'algorithm': 'TPPO', 'gamma': 0.999, 'lam': 0.986, 'device': 'CPU', 'actor': {'max_update_p': 0.081, 'min_update_p': 0.0004, 'nr_of_clauses': 1500, 'T': 1319, 's': 3.6800000000000024, 'y_max': 100, 'y_min': 0, 'bits_per_feature': 7, 'number_of_state_bits_ta': 3}, 'critic': {'max_update_p': 0.043, 'min_update_p': 0.0, 'nr_of_clauses': 1120, 'T': 380, 's': 8.240000000000006, 'y_max': 93, 'y_min': 0.1, 'bits_per_feature': 5, 'number_of_state_bits_ta': 6}, 'epochs': 4, 'test_freq': 10, 'save': True, 'seed': 42, 'threshold': 0, 'n_timesteps': 1432, 'dataset_file_name': 'cartpole_obs_data'}
config = {'env_name': "cartpole", 'algorithm': 'TPPO', 'gamma': 0.987, 'lam': 0.988, 'device': 'CPU', 'actor': {'max_update_p': 0.038, 'min_update_p': 0.0008, 'nr_of_clauses': 1460, 'T': 350, 's': 6.300000000000004, 'y_max': 100, 'y_min': 0, 'bits_per_feature': 7, 'number_of_state_bits_ta': 3}, 'critic': {'max_update_p': 0.056, 'min_update_p': 0.0, 'nr_of_clauses': 980, 'T': 940, 's': 6.280000000000005, 'y_max': 97, 'y_min': 0.4, 'bits_per_feature': 8, 'number_of_state_bits_ta': 6}, 'epochs': 8, 'test_freq': 100, 'save': True, 'seed': 42, 'threshold': 0, 'n_timesteps': 504, 'dataset_file_name': 'cartpole_obs_data'}


print(config)

env = gym.make("CartPole-v1")

agent = TPPO(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'],
                                         tms[i]['feedback_to_clauses'])
test_policy(save_file, agent.policy.actor, config['env_name'])
