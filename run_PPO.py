import torch
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from algorithms.Proximal_policy.PPO import PPO
from algorithms.policy.DNN import ActorCriticPolicy as Policy


#run 33 - 500 - config = {'algorithm': 'PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range':0.2, 'batch_size': 64, 'epochs': 1,'hidden_size': 32, 'learning_rate': 0.01, 'test_freq': 1, "save": False}
#run 37 - 500 - config = {'algorithm': 'PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.5, 'batch_size': 64, 'epochs': 1, 'hidden_size': 32, 'learning_rate': 0.015, 'test_freq': 1, "save": True}
#run 37 much more stable
#config = {'algorithm': 'PPO', 'gamma': 0.99, 'lam': 0.96, 'clip_range': 0.20, 'batch_size': 1000, 'epochs': 2, 'hidden_size': 256, 'learning_rate': 0.003, 'test_freq': 1, "save": True}
#config = {'algorithm': 'PPO', 'gamma': 0.925, 'lam': 0.943, 'clip_range': 0.03, 'batch_size': 336, 'epochs': 8, 'hidden_size': 32, 'learning_rate': 0.0055, 'test_freq': 1, "save": True}
#config = {'algorithm': 'PPO', 'gamma': 0.925, 'lam': 0.943, 'clip_range': 0.01, 'batch_size': 336, 'epochs': 8, 'hidden_size': 64, 'learning_rate': 0.0055, 'test_freq': 1, "save": True}
#config = {'algorithm': 'PPO', 'gamma': 0.925, 'lam': 0.943, 'clip_range': 0.005, 'batch_size': 336, 'epochs': 8, 'hidden_size': 64, 'learning_rate': 0.0001, 'test_freq': 1, "save": True}
#config = {'algorithm': 'PPO', 'gamma': 0.987, 'lam': 0.942, 'clip_range': 0.257, 'batch_size': 144, 'epochs': 6, 'hidden_size': 224, 'learning_rate': 0.0061, 'test_freq': 1, "save": True}
#config = {'env_name': 'cartpole', 'algorithm': 'PPO', 'n_steps': 256, 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.2, 'batch_size': 32, 'epochs': 8, 'hidden_size': 64, 'learning_rate': 0.001, 'test_freq': 1, "save": True}
config = {'env_name': 'cartpole', 'algorithm': 'PPO', 'n_steps': 1568, 'gamma': 0.938, 'lam': 0.973, 'clip_range': 0.35000000000000003, 'batch_size': 64, 'epochs': 3, 'hidden_size': 32, 'learning_rate': 0.0006600000000000001, 'test_freq': 1, 'save': True}
print(config)

env = gym.make("CartPole-v1")
#env = gym.make("Acrobot-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=2_500)

from test_policy import test_policy

file = f'./results/PPO/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'./results/PPO/{agent.run_id}/final_test_results', model.actor, config['env_name'])