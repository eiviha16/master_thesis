import numpy as np
import os
import yaml
from tqdm import tqdm
import random
from algorithms.misc.batch_buffer import Batch
from algorithms.misc.plot_test_results import plot_test_results
import torch
import torch.nn.functional as F


class PPO():
    def __init__(self, env, Policy):
        self.env = env
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]

        self.policy = Policy()
        self.batch = Batch()

    def calculate_advantage(self):
        advantage = 0
        next_value = 0
        for i in reversed(range(len(self.batch.actions))):
            dt = self.batch.rewards[i] + self.gamma * next_value - self.batch.value[i]
            advantage = dt + self.gamma * self.lam * advantage * int(not self.batch.dones[i])
            next_value = self.batch.value[i]

            self.batch.advantages.insert(0, advantage)
    def normalize_advantages(self):
        advantages = np.array(self.batch.advantages)
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.batch.advantages = norm_advantages


    def rollout(self):
        obs, _ = self.env.reset(seed=42)
        while True:
            action, action_log_prob, value = self.policy.actor.get_action(obs)
            obs, reward, done, truncated, _ = self.env.step(action)
            self.batch.save_experience(action, action_log_prob, value, obs, reward, done)
            if done or truncated:
                break
    def train(self):
        pass
    def learn(self, nr_of_episodes):
        for episode in range(nr_of_episodes):
            self.rollout()
            self.calculate_advantage()
            self.normalize_advantages()

            self.train()

            self.test()
            self.batch.clear()

    def test(self):
        # remember to remove exploration when doing this
        pass

    def test(self):
        pass
