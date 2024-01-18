import numpy as np
import random


class Batch:
    def __init__(self):
        self.actions = []
        self.action_log_prob = []
        self.value = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.advantages = []

    def clear(self):
        self.actions = []
        self.action_log_prob = []
        self.value = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.advantages = []

    def save_experience(self, action, action_log_prob, value, obs, reward, done):
        self.actions.append(action)
        self.action_log_prob = action_log_prob
        self.value = value
        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(done)


if __name__ == '__main__':
    pass
