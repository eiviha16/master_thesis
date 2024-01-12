import numpy as np
import torch
import random

random.seed(42)
np.random.seed(1)
torch.manual_seed(42)

import gymnasium as gym

def test_policy(policy):
    seeds = [np.random.randint(1, 100000000) for i in range(100)]
    env = gym.make("CartPole-v1")

    episode_rewards = np.array([0 for _ in range(100)])
    actions = 0
    _q_vals = [0, 0]
    for episode in range(100):
        obs, _ = env.reset(seed=seeds[episode])  # episode)
        #print(episode)
        while True:
            try:
                q_vals = policy.predict(obs)

                action = np.argmax(policy.predict(obs))
                actions += 1
                obs, reward, done, truncated, _ = env.step(action)
            except:
                action = torch.argmax(policy.predict(obs))
                obs, reward, done, truncated, _ = env.step(action.detach().numpy())
            episode_rewards[episode] += reward
            if done or truncated:
                break

    mean = np.mean(episode_rewards)
    std = np.std(episode_rewards)
    import os
    print(f'Mean reward: {mean}')
    print(f'Mean std: {std}')
    print(f'Actions: {actions}')
