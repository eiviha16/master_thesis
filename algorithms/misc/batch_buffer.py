import numpy as np
import random
import torch


class Batch:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.actions = []
        self.action_log_prob = []
        self.values = []
        self.next_values = []
        self.obs = []
        self.rewards = []
        self.terminated = []
        self.advantages = []
        self.entropies = []
        self.discounted_rewards = []
        self.trunc = []

        self.sampled_actions = []
        self.sampled_action_log_prob = []
        self.sampled_values = []
        self.sampled_next_values = []
        self.sampled_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []
        self.sampled_advantages = []
        self.sampled_entropies = []
        self.sampled_discounted_rewards = []
        self.sampled_trunc = []

    def clear(self):
        self.actions = []
        self.action_log_prob = []
        self.values = []
        self.next_values = []
        self.obs = []
        self.rewards = []
        self.terminated = []
        self.advantages = []
        self.entropies = []
        self.discounted_rewards = []
        self.trunc = []

    def shuffle(self):
        self.sampled_actions = []
        self.sampled_action_log_prob = []
        self.sampled_values = []
        self.sampled_next_values = []
        self.sampled_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []
        self.sampled_advantages = []
        self.sampled_entropies = []
        self.sampled_discounted_rewards = []
        self.sampled_trunc = []

        sample = random.sample(range(len(self.rewards)), len(self.rewards))
        for i, s in enumerate(sample):
            self.sampled_actions.append(self.actions[s])
            self.sampled_action_log_prob.append(self.action_log_prob[s])
            self.sampled_values.append(self.values[s])
            self.sampled_next_values.append(self.next_values[s])
            self.sampled_obs.append(self.obs[s])
            self.sampled_rewards.append(self.rewards[s])
            self.sampled_terminated.append(self.terminated[s])
            self.sampled_advantages.append(self.advantages[s])
            self.sampled_entropies.append(self.entropies[s])
            self.sampled_trunc.append(self.trunc[s])

            # self.sampled_discounted_rewards.append(self.discounted_rewards[s])

        self.sampled_actions = np.array(self.sampled_actions)
        self.sampled_action_log_prob = np.array(self.sampled_action_log_prob)
        self.sampled_values = np.array(self.sampled_values)
        self.sampled_next_values = np.array(self.sampled_next_values)
        self.sampled_obs = np.array(self.sampled_obs)
        self.sampled_rewards = np.array(self.sampled_rewards)
        self.sampled_terminated = np.array(self.sampled_terminated)
        self.sampled_advantages = np.array(self.sampled_advantages)
        self.sampled_entropies = np.array(self.sampled_entropies)
        self.sampled_trunc = np.array(self.trunc)

        # self.sampled_discounted_rewards = np.array(self.sampled_discounted_rewards)

    def sample(self):
        self.sampled_actions = []
        self.sampled_action_log_prob = []
        self.sampled_values = []
        self.sampled_next_values = []
        self.sampled_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []
        self.sampled_advantages = []
        self.sampled_entropies = []
        self.sampled_discounted_rewards = []
        self.trunc = []

        if len(self.terminated) > self.batch_size:
            sample = random.sample(range(len(self.rewards)), self.batch_size)
            for i, s in enumerate(sample):
                self.sampled_actions.append(self.actions[s])
                self.sampled_action_log_prob.append(self.action_log_prob[s])
                self.sampled_values.append(self.values[s])
                self.sampled_next_values.append(self.next_values[s])
                self.sampled_obs.append(self.obs[s])
                self.sampled_rewards.append(self.rewards[s])
                self.sampled_terminated.append(self.terminated[s])
                self.sampled_advantages.append(self.advantages[s])
                self.sampled_entropies.append(self.entropies[s])
                self.sampled_discounted_rewards.append(self.discounted_rewards[s])
                self.sampled_trunc.append(self.trunc[s])

            self.sampled_actions = np.array(self.sampled_actions)
            self.sampled_action_log_prob = np.array(self.sampled_action_log_prob)
            self.sampled_values = np.array(self.sampled_values)
            self.sampled_next_values = np.array(self.sampled_next_values)
            self.sampled_obs = np.array(self.sampled_obs)
            self.sampled_rewards = np.array(self.sampled_rewards)
            self.sampled_terminated = np.array(self.sampled_terminated)
            self.sampled_advantages = np.array(self.sampled_advantages)
            self.sampled_entropies = np.array(self.sampled_entropies)
            self.sampled_discounted_rewards = np.array(self.sampled_discounted_rewards)
            self.sampled_trunc = np.array(self.trunc)


        else:
            self.sampled_actions = self.actions
            self.sampled_action_log_prob = self.action_log_prob
            self.sampled_values = self.values
            self.sampled_next_values = self.next_values
            self.sampled_obs = self.obs
            self.sampled_rewards = self.rewards
            self.sampled_terminated = self.terminated
            self.sampled_advantages = self.advantages
            self.sampled_entropies = self.entropies
            self.sampled_discounted_rewards = self.discounted_rewards
            self.sampled_trunc = self.trunc

    def save_experience(self, action, action_log_prob, value, next_value, obs, reward, terminated, trunc, entropy):
        self.actions.append(action)
        self.action_log_prob.append(action_log_prob)
        self.values.append(value)
        self.next_values.append(next_value)
        # self.action_log_prob.append(action_log_prob.detach().numpy())
        # self.values.append(value.detach().numpy())
        self.obs.append(obs)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.entropies.append(entropy)
        self.trunc.append(trunc)

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.action_log_prob = np.array(self.action_log_prob)
        self.values = np.array(self.values)
        self.next_values = np.array(self.next_values)
        self.obs = np.array(self.obs)
        self.rewards = np.array(self.rewards)
        self.terminated = np.array(self.terminated)
        self.entropies = np.array(self.entropies)
        self.trunc = np.array(self.trunc)


class Batch_VPG:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.actions = []
        self.action_log_prob = []
        self.obs = []
        self.rewards = []
        self.terminated = []
        self.discounted_rewards = []

    def clear(self):
        self.actions = []
        self.action_log_prob = []
        self.obs = []
        self.rewards = []
        self.terminated = []
        self.discounted_rewards = []

    def save_experience(self, action, action_log_prob, obs, reward, terminated):
        self.actions.append(action)
        self.action_log_prob.append(action_log_prob)
        self.obs.append(obs)
        self.rewards.append(reward)
        self.terminated.append(terminated)

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.action_log_prob = np.array(self.action_log_prob)
        self.obs = np.array(self.obs)
        self.rewards = np.array(self.rewards)
        self.terminated = np.array(self.terminated)

    def convert_to_tensor(self):
        self.action_log_prob = torch.stack(self.action_log_prob)
        # self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
        self.discounted_rewards = torch.tensor(self.discounted_rewards, dtype=torch.float32)


class Batch_TM_VPG:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.actions = []
        self.action_log_prob = []
        self.cur_obs = []
        self.next_obs = []
        self.rewards = []
        self.q_vals = []
        self.terminated = []
        self.sampled_actions = []
        self.sampled_action_log_prob = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_q_vals = []
        self.sampled_terminated = []

    def sample(self):
        self.sampled_actions = []
        self.sampled_action_log_prob = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_q_vals = []
        self.sampled_rewards = []
        self.sampled_terminated = []

        if len(self.terminated) > self.batch_size:
            sample = random.sample(range(len(self.rewards)), self.batch_size)
            for i, s in enumerate(sample):
                self.sampled_actions.append(self.actions[s])
                self.sampled_action_log_prob.append(self.action_log_prob[s])
                self.sampled_cur_obs.append(self.cur_obs[s])
                self.sampled_next_obs.append(self.next_obs[s])
                self.sampled_q_vals.append(self.q_vals[s])
                self.sampled_rewards.append(self.rewards[s])
                self.sampled_terminated.append(self.terminated[s])

            self.sampled_actions = np.array(self.sampled_actions)
            self.sampled_action_log_prob = np.array(self.sampled_action_log_prob)
            self.sampled_cur_obs = np.array(self.sampled_cur_obs)
            self.sampled_next_obs = np.array(self.sampled_next_obs)
            self.sampled_q_vals = np.array(self.sampled_q_vals)
            self.sampled_rewards = np.array(self.sampled_rewards)
            self.sampled_terminated = np.array(self.sampled_terminated)

        else:
            self.sampled_actions = self.actions
            self.sampled_action_log_prob = self.action_log_prob
            self.sampled_cur_obs = self.cur_obs
            self.sampled_next_obs = self.next_obs
            self.sampled_q_vals = self.q_vals
            self.sampled_rewards = self.rewards
            self.sampled_terminated = self.terminated

    def clear(self):
        self.actions = []
        self.action_log_prob = []
        self.cur_obs = []
        self.next_obs = []
        self.rewards = []
        self.q_vals = []
        self.terminated = []
        self.discounted_rewards = []

    def save_experience(self, action, action_log_prob, q_val, cur_obs, next_obs, reward, terminated):
        self.actions.append(action)
        self.action_log_prob.append(action_log_prob)
        self.cur_obs.append(cur_obs)
        self.next_obs.append(next_obs)
        self.q_vals.append(q_val)
        self.rewards.append(reward)
        self.terminated.append(terminated)

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.action_log_prob = np.array(self.action_log_prob)
        self.cur_obs = np.array(self.cur_obs)
        self.next_obs = np.array(self.next_obs)
        self.q_vals = np.array(self.q_vals)
        self.rewards = np.array(self.rewards)
        self.terminated = np.array(self.terminated)


class Batch_TM_DDPG:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.actions = []
        self.cur_obs = []
        self.next_obs = []
        self.rewards = []
        self.terminated = []

        self.sampled_actions = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []

    def sample(self):
        self.sampled_actions = []
        self.sampled_cur_obs = []
        self.sampled_next_obs = []
        self.sampled_rewards = []
        self.sampled_terminated = []

        if len(self.terminated) > self.batch_size:
            sample = random.sample(range(len(self.rewards)), self.batch_size)
            for i, s in enumerate(sample):
                self.sampled_actions.append(self.actions[s])
                self.sampled_cur_obs.append(self.cur_obs[s])
                self.sampled_next_obs.append(self.next_obs[s])
                self.sampled_rewards.append(self.rewards[s])
                self.sampled_terminated.append(self.terminated[s])

            self.sampled_actions = np.array(self.sampled_actions)
            self.sampled_cur_obs = np.array(self.sampled_cur_obs)
            self.sampled_next_obs = np.array(self.sampled_next_obs)
            self.sampled_rewards = np.array(self.sampled_rewards)
            self.sampled_terminated = np.array(self.sampled_terminated)

        else:
            self.sampled_actions = self.actions
            self.sampled_cur_obs = self.cur_obs
            self.sampled_next_obs = self.next_obs
            self.sampled_rewards = self.rewards
            self.sampled_terminated = self.terminated

    def clear(self):
        self.actions = []
        self.cur_obs = []
        self.next_obs = []
        self.rewards = []
        self.terminated = []

    def save_experience(self, action, cur_obs, next_obs, reward, terminated):
        self.actions.append(action)
        self.cur_obs.append(cur_obs)
        self.next_obs.append(next_obs)
        self.rewards.append(reward)
        self.terminated.append(terminated)

    def convert_to_numpy(self):
        self.actions = np.array(self.actions)
        self.cur_obs = np.array(self.cur_obs)
        self.next_obs = np.array(self.next_obs)
        self.rewards = np.array(self.rewards)
        self.terminated = np.array(self.terminated)


if __name__ == '__main__':
    pass
