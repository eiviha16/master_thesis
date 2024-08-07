import numpy as np
import random


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, n=-1):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.actions = []  # 0 for i in range(buffer_size)]
        self.cur_obs = []  # 0 for i in range(buffer_size)]
        self.next_obs = []  # 0 for i in range(buffer_size)]
        self.rewards = []  # 0 for i in range(buffer_size)]
        self.terminated = []  # 0 for i in range(buffer_size)]
        self.trunc = []

        self._actions = [[], []]  # 0 for i in range(buffer_size)]
        self._cur_obs = [[], []]  # 0 for i in range(buffer_size)]
        self._next_obs = [[], []]  # 0 for i in range(buffer_size)]
        self._rewards = [[], []]  # 0 for i in range(buffer_size)]
        self._terminated = [[], []]  # 0 for i in range(buffer_size)]

        self.sampled_actions = []  # 0 for i in range(batch_size)]
        self.sampled_cur_obs = []  # 0 for i in range(batch_size)]
        self.sampled_next_obs = []  # 0 for i in range(batch_size)]
        self.sampled_rewards = []  # 0 for i in range(batch_size)]
        self.sampled_terminated = []  # 0 for i in range(batch_size)]
        self.sampled_trunc = []

        self.indices = []  # i for i in range(buffer_size)]
        self.n = n

    def clear_cache(self):
        self.sampled_actions = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []
        self.sampled_trunc = []


    def sample(self):
        self.clear_cache()
        sample = random.sample(range(len(self.rewards)), self.batch_size)
        for i, s in enumerate(sample):
            self.sampled_actions.append(self.actions[s])
            self.sampled_cur_obs.append(self.cur_obs[s])
            self.sampled_next_obs.append(self.next_obs[s])
            self.sampled_rewards.append(self.rewards[s])
            self.sampled_terminated.append(self.terminated[s])
            self.sampled_trunc.append(self.trunc[s])

    def sample_n_seq(self):
        self.clear_cache()
        sample = random.sample(range(len(self.rewards) - self.n), self.batch_size)
        for i, s in enumerate(sample):
            self.sampled_actions.append(self.actions[s: s + self.n])
            self.sampled_cur_obs.append(self.cur_obs[s: s + self.n])
            self.sampled_next_obs.append(self.next_obs[s: s + self.n])
            self.sampled_rewards.append(self.rewards[s: s + self.n])
            self.sampled_terminated.append(self.terminated[s: s + self.n])
            self.sampled_trunc.append(self.trunc[s: s+self.n])

    def save_experience(self, action, cur_obs, next_obs, reward, terminated, trunc):
        if self.buffer_size <= len(self.rewards):
            self.actions.pop(0)
            self.cur_obs.pop(0)
            self.next_obs.pop(0)
            self.rewards.pop(0)
            self.terminated.pop(0)
            self.trunc.pop(0)

        self.actions.append(action)
        self.cur_obs.append(cur_obs)
        self.next_obs.append(next_obs)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.trunc.append(trunc)


if __name__ == '__main__':
    pass
