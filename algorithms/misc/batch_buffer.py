import numpy as np
import random
import torch

class Batch:
    def __init__(self):
        self.actions = []
        self.action_log_prob = []
        self.values = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.advantages = []

    def clear(self):
        self.actions = []
        self.action_log_prob = []
        self.values = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.advantages = []

    def save_experience(self, action, action_log_prob, value, obs, reward, done):
        self.actions.append(action)
        self.action_log_prob.append(action_log_prob.detach().numpy())
        self.values.append(value.detach().numpy())
        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(int(done))

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.action_log_prob = np.array(self.action_log_prob)
        self.values = np.array(self.values)
        self.obs = np.array(self.obs)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)


if __name__ == '__main__':
    pass
