import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Tsetlin_Actor_Critic.TAAC import TAAC
from algorithms.policy.TAAC_policy import AdvantageActorCriticPolicy as Policy

"""actor = {'nr_of_clauses': 1250, 'T': int(1250 * 0.34), 's': 1.76, 'bits_per_feature': 6, 'number_of_state_bits_ta': 6}

critic = {"max_update_p": 0.075, "min_update_p": 0.0, 'nr_of_clauses': 1340, 'T': int(1240 * 0.78), 's': 1.65,
          'y_max': 100.0, 'y_min': 0, 'bits_per_feature': 4, 'number_of_state_bits_ta': 7}
config = {'env_name': "cartpole", 'algorithm': 'TPPO', "n_timesteps": 512, 'gamma': 0.99, 'lam': 0.96, "actor": actor,
          "critic": critic, 'device': 'CPU', 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42,
          "dataset_file_name": "cartpole_obs_data",
          "threshold": -1000}  # "dataset_file_name": "acrobot_obs_data"}"observation_data"""

config = {'env_name': "cartpole", 'algorithm': 'TAAC', 'gamma': 1, 'lam': 0.97, 'device': 'CPU', 'actor': {'nr_of_clauses': 1950, 'T': 39, 's': 3.450000000000002, 'device': 'CPU', 'bits_per_feature': 6, 'seed': 42, 'number_of_state_bits_ta': 4}, 'critic': {'max_update_p': 0.482, 'min_update_p': 0.0, 'nr_of_clauses': 1010, 'T': 383, 's': 9.000000000000007, 'y_max': 99, 'y_min': 0.7000000000000001, 'bits_per_feature': 4, 'number_of_state_bits_ta': 4}, 'epochs': 3, 'test_freq': 5, 'save': True, 'seed': 42, 'threshold': 0, 'n_timesteps': 672, 'dataset_file_name': 'cartpole_obs_data'}

print(config)

env = gym.make("CartPole-v1")

agent = TAAC(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy

save_file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results'

tms = torch.load(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best')

for i in range(len(tms)):
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'],
                                         tms[i]['feedback_to_clauses'])
test_policy(save_file, agent.policy.actor, config['env_name'])
