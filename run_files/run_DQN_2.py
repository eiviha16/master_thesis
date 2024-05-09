import gymnasium as gym
import random
from algorithms.Q_Networks.DQN import DQN
from algorithms.policy.DNN import Policy
import torch
import numpy as np

config = {'env_name': 'cartpole', 'algorithm': 'DQN', 'gamma': 0.99, 'epsilon_init': 1.0, 'epsilon_decay': 0.001, "epsilon_min":0.01, 'buffer_size': 50_000, 'batch_size': 16, 'hidden_size': 80, 'learning_rate': 0.0003, 'test_freq': 1, 'threshold_score': 450, "save": True}
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