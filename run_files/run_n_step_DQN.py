import random
import torch
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from algorithms.Q_Network.n_step_DQN import DQN
from algorithms.policy.DNN import Policy
import gymnasium as gym


#config = {'algorithm': 'DQN', 'gamma': 0.98, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 20000, 'batch_size': 256, 'epochs': 4, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
#config = {'algorithm': 'DQN', 'gamma': 0.99, 'c': 30, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 20000, 'batch_size': 256, 'epochs': 4, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
#config = {"env_name": "acrobot", 'algorithm': 'n_step_DQN', 'n_steps': 47, 'gamma': 0.99, 'c':  1, 'exploration_prob_init': 1.0, 'exploration_prob_decay': 0.001, 'buffer_size': 5000, 'batch_size': 64, 'epochs': 4, 'hidden_size': 128, 'learning_rate': 0.0003, 'test_freq': 50, "save": True}


config = {'env_name': 'acrobot', 'n_steps': 28, 'algorithm': 'n_step_DQN', 'c': 0, 'gamma': 0.994, 'buffer_size': 7000, 'batch_size': 96, 'exploration_prob_init': 0.7, 'exploration_prob_decay': 0.005, 'epochs': 8, 'hidden_size': 160, 'learning_rate': 0.00623, 'test_freq': 50, 'save': True}

#env = gym.make("CartPole-v1")
env = gym.make("Acrobot-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)

from test_policy import test_policy
file = f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'../results/{config["env_name"]}/{config["algorithm"]}/{agent.run_id}/final_test_results', model, config["env_name"])