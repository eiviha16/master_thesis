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