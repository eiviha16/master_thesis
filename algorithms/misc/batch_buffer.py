import numpy as np
import random
import torch


class Batch:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.actions = []
        self.action_log_prob = []
        self.values = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.entropies = []
        self.discounted_rewards = []

        self.sampled_actions = []
        self.sampled_action_log_prob = []
        self.sampled_values = []
        self.sampled_obs = []
        self.sampled_rewards = []
        self.sampled_dones = []
        self.sampled_advantages = []
        self.sampled_entropies = []
        self.sampled_discounted_rewards = []

    def clear(self):
        self.actions = []
        self.action_log_prob = []
        self.values = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.entropies = []
        self.discounted_rewards = []

    def sample(self):
        self.sampled_actions = []
        self.sampled_action_log_prob = []
        self.sampled_values = []
        self.sampled_obs = []
        self.sampled_rewards = []
        self.sampled_dones = []
        self.sampled_advantages = []
        self.sampled_entropies = []
        self.sampled_discounted_rewards = []

        if len(self.dones) > self.batch_size:
            sample = random.sample(range(len(self.rewards)), self.batch_size)
            for i, s in enumerate(sample):
                self.sampled_actions.append(self.actions[s])
                self.sampled_action_log_prob.append(self.action_log_prob[s])
                self.sampled_values.append(self.values[s])
                self.sampled_obs.append(self.obs[s])
                self.sampled_rewards.append(self.rewards[s])
                self.sampled_dones.append(self.dones[s])
                self.sampled_advantages.append(self.advantages[s])
                self.sampled_entropies.append(self.entropies[s])
                self.sampled_discounted_rewards.append(self.discounted_rewards[s])

            self.sampled_actions = np.array(self.sampled_actions)
            self.sampled_action_log_prob = np.array(self.sampled_action_log_prob)
            self.sampled_values = np.array(self.sampled_values)
            self.sampled_obs = np.array(self.sampled_obs)
            self.sampled_rewards = np.array(self.sampled_rewards)
            self.sampled_dones = np.array(self.sampled_dones)
            self.sampled_advantages = np.array(self.sampled_advantages)
            self.sampled_entropies = np.array(self.sampled_entropies)
            self.sampled_discounted_rewards = np.array(self.sampled_discounted_rewards)

        else:
            self.sampled_actions = self.actions
            self.sampled_action_log_prob = self.action_log_prob
            self.sampled_values = self.values
            self.sampled_obs = self.obs
            self.sampled_rewards = self.rewards
            self.sampled_dones = self.dones
            self.sampled_advantages = self.advantages
            self.sampled_entropies = self.entropies
            self.sampled_discounted_rewards = self.discounted_rewards

    def save_experience(self, action, action_log_prob, value, obs, reward, done, entropy=0):
        self.actions.append(action)
        # self.action_log_prob.append(action_log_prob)
        # self.values.append(value)
        self.action_log_prob.append(action_log_prob.detach().numpy())
        self.values.append(value.detach().numpy())
        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(done)
        # self.dones.append(int(done))
        self.entropies.append(entropy)

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.action_log_prob = np.array(self.action_log_prob)
        self.values = np.array(self.values)
        self.obs = np.array(self.obs)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)
        self.entropies = np.array(self.entropies)


class Batch_VPG:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.actions = []
        self.action_log_prob = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.discounted_rewards = []

    def clear(self):
        self.actions = []
        self.action_log_prob = []
        self.obs = []
        self.rewards = []
        self.dones = []
        self.discounted_rewards = []

    def save_experience(self, action, action_log_prob, obs, reward, done):
        self.actions.append(action)
        self.action_log_prob.append(action_log_prob)
        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(done)

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.action_log_prob = np.array(self.action_log_prob)
        self.obs = np.array(self.obs)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)

    def convert_to_tensor(self):
        self.action_log_prob = torch.stack(self.action_log_prob)
        # self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
        self.discounted_rewards = torch.tensor(self.discounted_rewards, dtype=torch.float32)


if __name__ == '__main__':
    pass

"""    



"""
"""
    def save_experience(self, action, action_log_prob, value, obs, reward, done):
        self.actions.append(action)
        #self.action_log_prob.append(action_log_prob)
        #self.values.append(value)
        self.action_log_prob.append(action_log_prob.detach().numpy())
        self.values.append(value.detach().numpy())
        self.obs.append(obs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.dones.append(int(done))

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.action_log_prob = np.array(self.action_log_prob)
        self.values = np.array(self.values)
        self.obs = np.array(self.obs)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)
"""
