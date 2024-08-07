import numpy as np
import os
import yaml
from tqdm import tqdm
import random
from algorithms.misc.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F


class DQN:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n
        self.obs_space_size = env.observation_space.shape[0]
        config['action_space_size'] = self.action_space_size
        config['obs_space_size'] = self.obs_space_size
        self.policy = Policy(self.obs_space_size, self.action_space_size, config)
        self.gamma = config['gamma']  # discount factor
        self.init_epsilon = config['epsilon_init']
        self.epsilon = self.init_epsilon
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.sampling_iterations = config['sampling_iterations']
        self.config = config
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 100
        self.save = config['save']

        if self.save:
            self.run_id = 'run_' + str(len([i for i in os.listdir(f"../results/{config['env_name']}/{config['algorithm']}")]) + 1)
            self.make_run_dir()
            self.save_config()
        else:
            self.run_id = "Unidentified run"
        self.has_reached_threshold = False
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081, 71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]
        self.best_scores = {'mean': -float('inf'), 'std': float('inf')}
        self.config = config
        self.q_values = {f'q{i}': [] for i in range(self.action_space_size)}
        self.nr_actions = 0

        self.announce()
        self.cur_episode = 0
        self.nr_of_steps = 0
        self.scores = []


    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def make_run_dir(self):
        if self.save:
            base_dir = '../results'
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            if not os.path.exists(os.path.join(base_dir, self.config['env_name'])):
                os.makedirs(os.path.join(base_dir, self.config['env_name']))
            if not os.path.exists(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'])):
                os.makedirs(os.path.join(base_dir, self.config['env_name'], self.config['algorithm']))
            if not os.path.exists(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id)):
                os.makedirs(os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id))
            self.save_path = os.path.join(base_dir, self.config['env_name'], self.config['algorithm'], self.run_id)

    def save_config(self):
        with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def temporal_difference(self, next_q_vals):
        return torch.tensor(self.replay_buffer.sampled_rewards) + (
                1 - torch.tensor(self.replay_buffer.sampled_terminated)) * self.gamma * next_q_vals

    def get_next_action(self, cur_obs):
        if np.random.random() < self.epsilon:
            q_vals = torch.tensor([np.random.random() for _ in range(self.action_space_size)])
        else:
            q_vals = self.policy.predict(cur_obs)
            for i, key in enumerate(self.q_values):
                self.q_values[key].append(q_vals[i])
        return torch.argmax(q_vals), q_vals

    def update_epsilon_greedy(self):
        self.epsilon = self.epsilon_min + (self.init_epsilon - self.epsilon_min) * np.exp(-self.cur_episode * self.epsilon_decay)


    def get_q_val_for_action(self, q_vals):
        sampled_actions = np.array(self.replay_buffer.sampled_actions)
        indices = sampled_actions[:, 0]
        selected_q_vals = q_vals[range(q_vals.shape[0]), indices]
        return selected_q_vals


    def n_step_temporal_difference(self, next_q_vals):
        target_q_vals = []
        for i in range(len(self.replay_buffer.sampled_rewards)):
            target_q_val = 0
            for j in range(len(self.replay_buffer.sampled_rewards[i])):
                target_q_val += (self.gamma ** j) * self.replay_buffer.sampled_rewards[i][j]
                if self.replay_buffer.sampled_terminated[i][j] or self.replay_buffer.sampled_trunc[i][j]:
                    break
            target_q_val += (1 - self.replay_buffer.sampled_terminated[i][j]) * (self.gamma ** j) * next_q_vals[i]
            target_q_vals.append(target_q_val)
        return torch.stack(target_q_vals)

    def train(self):
        for _ in range(self.sampling_iterations):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample_n_seq()
            with torch.no_grad():
                sampled_next_obs = np.array(self.replay_buffer.sampled_next_obs)
                next_q_vals = self.policy.predict(sampled_next_obs[:, -1, :])
                next_q_vals, _ = torch.max(next_q_vals, dim=1)
                target_q_vals = self.n_step_temporal_difference(next_q_vals)
            sampled_cur_obs = np.array(self.replay_buffer.sampled_cur_obs)
            cur_q_vals = self.policy.predict(sampled_cur_obs[:, 0, :])
            cur_q_vals = self.get_q_val_for_action(cur_q_vals)
            self.policy.optimizer.zero_grad()
            loss = F.mse_loss(target_q_vals, cur_q_vals)
            loss.backward()
            self.policy.optimizer.step()


    def rollout(self):
        cur_obs, _ = self.env.reset(seed=random.randint(1, 100))
        while True:
            action, _ = self.get_next_action(cur_obs)
            action = action.numpy()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(terminated), truncated)
            cur_obs = next_obs
            self.nr_of_steps += 1
            if terminated or truncated:
                break
            #if self.nr_of_steps > 1_000 and self.nr_of_steps - self.config['n_steps'] >= self.batch_size:
            #    self.train()


    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            self.cur_episode = episode + 1
            if episode % self.test_freq == 0:
                self.test(self.nr_of_steps)
            self.rollout()
            if self.nr_of_steps - self.config['n_steps'] >= self.batch_size:
                self.train()
            self.update_epsilon_greedy()


    def test(self, nr_of_steps):
        episode_rewards = np.array([0 for i in range(self.nr_of_test_episodes)])
        for episode in range(self.nr_of_test_episodes):
            for q_val in self.q_values:
                self.q_values[q_val] = []
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])
            while True:
                q_vals = self.policy.predict(obs)
                action = torch.argmax(q_vals)
                obs, reward, terminated, truncated, _ = self.env.step(action.numpy())
                episode_rewards[episode] += reward
                if terminated or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.scores.append(mean)

        self.save_results(mean, std, nr_of_steps)
        if mean > self.best_scores['mean']:
            self.save_model('best_model')
            self.best_scores['mean'] = mean
            print(f'New best mean after {nr_of_steps} steps: {mean}!')
        self.save_model('last_model')



    def save_model(self, file_name):
        if self.save:
            torch.save(self.policy, os.path.join(self.save_path, file_name))

    def save_results(self, mean, std, nr_of_steps):
        if self.save:
            file_name = 'test_results.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, file_name))

            with open(os.path.join(self.save_path, file_name), "a") as file:
                if not file_exists:
                    file.write("mean,std,steps\n")
                file.write(f"{mean},{std},{nr_of_steps}\n")


