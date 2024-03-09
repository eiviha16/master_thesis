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
config = {'algorithm': 'PPO', 'gamma': 0.99, 'lam': 0.95, 'clip_range': 0.2, 'batch_size': 64, 'epochs': 4, 'hidden_size': 32, 'learning_rate': 0.003, 'test_freq': 1, "save": True}

print(config)

env = gym.make("CartPole-v1")


agent = PPO(env, Policy, config)
agent.learn(nr_of_episodes=10_000)

from test_policy import test_policy

file = f'./results/PPO/{agent.run_id}/best_model'
model = torch.load(file)
test_policy(f'./results/PPO/{agent.run_id}/final_test_results', model.actor)