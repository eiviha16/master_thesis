import numpy as np
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Q_Network.n_step_TMQN import TMQN
from algorithms.policy.RTM import Policy


#Winner run 76 - 500.0 - 500.0 - config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': True, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1, 'threshold_score': 450, "save": True, "seed": 42, "balance_feedback": False, "min_feedback_p": 1.0, 'dynamic_memory': False, 'dynamic_memory_max_size': 10, 'number_of_state_bits_ta': 10}
#run 94 - 498.95 - 7.41 ,config = {'algorithm': 'n_step_TMQN', 'n_steps': 5, 'nr_of_clauses': 1000, 'T': 100, 's': 4.9, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}
config = {'algorithm': 'n_step_TMQN', 'n_steps': 10, 'nr_of_clauses': 1000, 'T': 350, 's': 3.7, 'y_max': 100, 'y_min': 0, 'device': 'CPU', 'weighted_clauses': False, 'bits_per_feature': 5, 'gamma': 0.95, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 2000, 'batch_size': 64, 'epochs': 2, 'test_freq': 1,  "save": True, 'dynamic_memory': False, 'number_of_state_bits_ta': 10}

env = gym.make("CartPole-v1")

agent = TMQN(env, Policy, config)
# agent.learn(nr_of_episodes=10000)
agent.learn(nr_of_episodes=2)

from test_policy import test_policy

#test_policy(agent.policy)
#test_policy(agent.current_policy)
save_file = f'results/n_step_TMQN/{agent.run_id}/final_test_results'

#agent.policy.tm1.set_state()
#agent.policy.tm2.set_state()

save_file = f'results/n_step_TMQN/{agent.run_id}'

tms = torch.load(f'results/n_step_TMQN/{agent.run_id}/best')

agent.policy.tm1.set_params(tms[0]['ta_state'], tms[0]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])
agent.policy.tm2.set_params(tms[1]['ta_state'], tms[1]['clause_sign'], tms[0]['clause_output'], tms[0]['feedback_to_clauses'])

test_policy(save_file, agent.policy)
