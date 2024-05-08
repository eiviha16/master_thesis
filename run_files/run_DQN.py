import gymnasium as gym
import random
from algorithms.Q_Network.DQN import DQN
from algorithms.policy.DNN import Policy
import torch
import numpy as np

config = {'env_name': 'cartpole', 'algorithm': 'DQN', 'gamma': 0.99, 'c':  1, 'epsilon_init': 1.0, 'epsilon_decay': 0.001, 'buffer_size': 40_000, 'batch_size': 256, 'epochs': 8, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, 'threshold_score': 450, "save": True}
print(config)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

env = gym.make("CartPole-v1")
#env = gym.make("Acrobot-v1")


agent = DQN(env, Policy, config)
agent.learn(nr_of_episodes=5000)
from test_policy import test_policy

file = f'../results/cartpole/DQN/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'../results/cartpole/DQN/{agent.run_id}/final_test_results', model,config['env_name'])