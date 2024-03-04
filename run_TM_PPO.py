import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.TM_PPO import PPO
from algorithms.policy.RTM import ActorCriticPolicy as Policy

#498.91 (on validation) so far run 28 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, "min_feedback_p": 1.0, 'number_of_state_bits_ta': 8}
#experiment with specificity?
#shows greater stability config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 6}
#498.12 - run 46 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 2.0, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 8}
#495.24 - run 48 - config = {'algorithm': 'TM_PPO', 'gamma': 0.99, 'lam': 0.95, 'nr_of_clauses': 1000, 'T': 250, 's': 3.7, 'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 8}
"""wandb: 	bits_per_feature: 12
wandb: 	clip: 0.3809999999999999
wandb: 	gamma: 0.944
wandb: 	lam: 0.948
wandb: 	nr_of_clauses: 1178
wandb: 	number_of_state_bits_ta: 4
wandb: 	specificity: 1.6300000000000006
wandb: 	t: 0.7"""
"""andb: Agent Starting Run: cfdh3rcq with config:
wandb: 	bits_per_feature: 12
wandb: 	clip: 0.001
wandb: 	gamma: 0.951
wandb: 	lam: 0.958
wandb: 	nr_of_clauses: 1160
wandb: 	number_of_state_bits_ta: 3
wandb: 	specificity: 1.5000000000000004
wandb: 	t: 0.76"""
#config = {'algorithm': 'TM_PPO', 'gamma': 0.94, 'lam': 0.946, "clip": 0.431, 'nr_of_clauses': 1050, 'T': int(1050 * 0.59), 's': 1.6, 'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 14,  'batch_size': 64, 'epochs': 1, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 6}
config = {'algorithm': 'TM_PPO', 'gamma': 0.952, 'lam': 0.964, "clip": 0.121, 'nr_of_clauses': 1020, 'T': int(1020 * 0.38), 's': 2.22, 'y_max': 7.5, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 7,  'batch_size': 64, 'epochs': 3, 'test_freq': 1, "save": True, "seed": 42, 'number_of_state_bits_ta': 4}
#change gamma and lambda

print(config)

env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

#agent.policy.actor.tms[0].set_state()
#agent.policy.actor.tms[1].set_state()
save_file = f'results/TM_PPO/{agent.run_id}'

tms = torch.load(f'results/TM_PPO/{agent.run_id}/best')

for i in range(len(tms)):
    #eval_ta_state, eval_clause_sign, eval_clause_output, eval_feedback_to_clauses
    agent.policy.actor.tms[i].set_params(tms[i]['ta_state'], tms[i]['clause_sign'], tms[i]['clause_output'], tms[i]['feedback_to_clauses'])

test_policy(save_file, agent.policy.actor)