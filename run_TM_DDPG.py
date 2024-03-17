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
actor = {'nr_of_classes': 2, 'nr_of_clauses': 1160, 'T': int(1160 * 0.8), 's': 1.43, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 3}
#actor = {'nr_of_classes': 2, 'nr_of_clauses': 1060, 'T': int(1060 * 0.2), 's': 2.54, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 9}
#critic = {'nr_of_clauses': 1150, 'T': int(1150 * 0.54), 's': 6.34, 'y_max': 65, 'y_min': 30, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7, "seed": 42, 'number_of_state_bits_ta': 8}
critic = {'nr_of_clauses': 1000, 'T': int(1000 * 0.27), 's': 5.42, 'y_max': 60, 'y_min': 35, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 12, "seed": 42, 'number_of_state_bits_ta': 3}
#critic = {'nr_of_clauses': 1900, 'T': int(1900 * 0.19), 's': 5.91, 'y_max': 65, 'y_min': 25, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 8, "seed": 42, 'number_of_state_bits_ta': 8}
#config = {'algorithm': 'TM_DDPG_2', 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.906, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 9352, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'gamma': 0.998, 'update_grad': -1, 'update_freq': 4, 'actor': actor, 'critic': critic, 'batch_size': 112, 'epochs': 7, 'test_freq': 1, "save": True}
#config = {'algorithm': 'TM_DDPG_2', 'buffer_size': 7092, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'soft_update_type': 'soft_update_2', 'gamma': 0.913, 'update_grad': -1, 'update_freq': 7, 'actor': actor, 'critic': critic, 'batch_size': 16, 'epochs': 2, 'test_freq': 1, "save": True}
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

"""import torch
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
config = {'algorithm': 'TM_DDPG', 'soft_update_type': 'soft_update_2', 'update_freq': 6, 'gamma': 0.98, 'actor': actor, 'critic': critic, 'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True}
#run 5 without initialization
#run 6 with initialization
print(config)

env = gym.make("CartPole-v1")


agent = VPG(env, Policy, config)
agent.learn(nr_of_episodes=10_0
            )

from test_policy import test_policy

tms = torch.load(f'results/TM_DDPG/{agent.run_id}/best')
for i in range(len(tms)):
    #eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

save_file = f'results/TM_DDPG/{agent.run_id}/final_test_results'

test_policy(save_file, agent.policy.actor)"""

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