import gymnasium as gym
import random
from algorithms.Q_Network.n_step_DQN import DQN
from algorithms.policy.DNN import Policy
import torch
import numpy as np

#config = {'algorithm': 'DQN', 'gamma': 0.98, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 20000, 'batch_size': 256, 'epochs': 4, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
#config = {'algorithm': 'DQN', 'gamma': 0.99, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 20000, 'batch_size': 256, 'epochs': 4, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
config = {'algorithm': 'n_step_DQN', 'n_step_TD': 20, 'gamma': 0.98, 'c':  1, 'exploration_prob_init': 0.5, 'exploration_prob_decay': 0.01, 'buffer_size': 50_000, 'batch_size': 256, 'epochs': 8, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
#config = {'algorithm': 'n_step_DQN', "n_steps": 10, 'gamma': 0.99, 'c':  1, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 50_000, 'batch_size': 256, 'epochs': 8, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=10000)
from test_policy import test_policy

file = f'./results/DQN/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'./results/DQN/{agent.run_id}/final_test_results', model.actor)