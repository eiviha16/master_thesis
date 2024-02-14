import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.VPG.TM_VPG import VPG
from algorithms.policy.RTM import ActorCriticPolicy2 as Policy

actor = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
critic = {'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 100.0, 'y_min': 20, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, "seed": 42, 'number_of_state_bits_ta': 8}
config = {'algorithm': 'TM_VPG', 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.98, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}

print(config)

env = gym.make("CartPole-v1")


agent = VPG(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

agent.policy.actor.tms[0].set_state()
agent.policy.actor.tms[1].set_state()
save_file = f'results/TM_VPG/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor)
