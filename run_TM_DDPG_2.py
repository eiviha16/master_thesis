import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.VPG.TM_DDPG import DDPG
from algorithms.policy.CTM import ActorCriticPolicy as Policy

actor = {'nr_of_classes': 2, 'nr_of_clauses': 1160, 'T': int(1160 * 0.52), 's': 4.5, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 9}
critic = {'nr_of_clauses': 1150, 'T': int(1150 * 0.54), 's': 6.34, 'y_max': 65, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 8}
config = {'algorithm': 'TM_DDPG_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.906, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
#run 5 without initialization
#run 6 with initialization
print(config)

env = gym.make("CartPole-v1")


agent = DDPG(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

tm = torch.load(f'results/TM_DDPG_2/{agent.run_id}/best')
agent.policy.actor.tm.set_params(tm[0]['ta_state'], tm[0]['clause_sign'], tm[0]['clause_count'])

save_file = f'results/TM_DDPG_2/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor)

"""
a_bits_per_feature
15
a_nr_of_clauses
868
a_number_of_state_bits_ta
3
a_specificity
6.67182600882225
a_t
0.7850247250916171
c_bits_per_feature
6
c_nr_of_clauses
1,244
c_number_of_state_bits_ta
4
c_specificity
6.614514432052328
c_t
0.03676095541541863
c_y_max
76
c_y_min
22
gamma
0.6706117521679655
update_freq
7
"""