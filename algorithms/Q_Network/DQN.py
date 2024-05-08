import numpy as np
import os
import yaml
from tqdm import tqdm
from algorithms.misc.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import random

class DQN:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n
        self.obs_space_size = env.observation_space.shape[0]
        self.policy = Policy(self.obs_space_size, self.action_space_size, config)
        self.gamma = config['gamma']  # discount factor
        self.epsilon = config['epsilon_init']
        self.epsilon_decay = config['epsilon_decay']

        self.sampling_iterations = config['sampling_iterations']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 100
        self.save = config['save']
        self.config = config
        self.save_path = ''

        if self.save:
            self.run_id = 'run_' + str(len([i for i in os.listdir(f"../results/{config['env_name']}/{config['algorithm']}")]) + 1)
            self.make_run_dir(config['algorithm'])
            self.save_config()
        else:
            self.run_id = "Unidentified run"
        self.has_reached_threshold = False
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081, 71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]
        self.best_scores = {'mean': -float('inf'), 'std': float('inf')}

        self.q_values = {f'q{i}': [] for i in range(self.action_space_size)}
        self.nr_actions = 0

        #self.observations = []
        self.announce()
        self.cur_episode = 0
        self.nr_of_steps = 0
        self.scores = []

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def make_run_dir(self, algorithm):
        if self.save:
            base_dir = '../results'
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            if not os.path.exists(os.path.join(base_dir, self.config['env_name'])):
                os.makedirs(os.path.join(base_dir, self.config['env_name']))
            if not os.path.exists(os.path.join(base_dir, self.config['env_name'], algorithm)):
                os.makedirs(os.path.join(base_dir, self.config['env_name'], algorithm))
            if not os.path.exists(os.path.join(base_dir, self.config['env_name'], algorithm, self.run_id)):
                os.makedirs(os.path.join(base_dir, self.config['env_name'], algorithm, self.run_id))
            self.save_path = os.path.join(base_dir, self.config['env_name'], algorithm, self.run_id)

    def save_config(self):
        if self.save:
            with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
                yaml.dump(self.config, yaml_file, default_flow_style=False)

    def temporal_difference(self, next_q_vals):
        return torch.tensor(self.replay_buffer.sampled_rewards) + (
                1 - torch.tensor(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals

    def get_next_action(self, cur_obs):
        if np.random.random() < self.epsilon:
            q_vals = torch.tensor([np.random.random() for _ in range(self.action_space_size)])
        else:
            q_vals = self.policy.predict(cur_obs)
            for i, key in enumerate(self.q_values):
                self.q_values[key].append(q_vals[i])
        return torch.argmax(q_vals), q_vals

    def update_epsilon_greedy(self):
        self.epsilon *= np.exp(-self.epsilon_decay)

    def get_q_val_for_action(self, q_vals):
        indices = np.array(self.replay_buffer.sampled_actions)
        selected_q_vals = q_vals[range(q_vals.shape[0]), indices]
        return selected_q_vals

    def train(self):
        for _ in range(self.sampling_iterations):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample()
            with torch.no_grad():
                next_q_vals = self.policy.predict(self.replay_buffer.sampled_next_obs)  # next_obs?
                next_q_vals, _ = torch.max(next_q_vals, dim=1)
                target_q_vals = self.temporal_difference(next_q_vals)

            cur_q_vals = self.policy.predict(self.replay_buffer.sampled_cur_obs)
            cur_q_vals = self.get_q_val_for_action(cur_q_vals)
            self.policy.optimizer.zero_grad()
            loss = F.mse_loss(target_q_vals, cur_q_vals)
            loss.backward()
            self.policy.optimizer.step()
    def rollout(self):
        cur_obs, _ = self.env.reset(seed=random.randint(1, 100))
        while True:
            action, _ = self.get_next_action(cur_obs)
            action = action.detach().numpy()
            next_obs, reward, done, truncated, _ = self.env.step(action)
            self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(done), self.nr_of_steps)
            cur_obs = next_obs
            self.nr_of_steps += 1
            if done or truncated:
                break
            if self.nr_of_steps >= self.batch_size:
                self.train()
    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            self.cur_episode = episode
            if episode % self.test_freq == 0:
                self.test(self.nr_of_steps)
            self.rollout()
            #if self.nr_of_steps >= self.batch_size:
            #    self.train()
            self.update_epsilon_greedy()


    def test(self, nr_of_steps):
        epsilon = self.epsilon
        self.epsilon = 0
        episode_rewards = np.array([0 for i in range(self.nr_of_test_episodes)])
        for episode in range(self.nr_of_test_episodes):
            for q_val in self.q_values:
                self.q_values[q_val] = []
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])
            while True:
                action, q_vals_ = self.get_next_action(obs)#.numpy()#.cpu().numpy()
                action = action.detach().numpy()
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.scores.append(mean)
        self.save_results(mean, std, nr_of_steps)
        self.epsilon = epsilon
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

    def save_q_vals(self, q_vals):
        if self.save:
            folder_name = 'q_values'
            file_name = f'{self.cur_episode}.csv'
            if not os.path.exists(os.path.join(self.save_path, folder_name)):
                os.makedirs(os.path.join(self.save_path, folder_name))
            file_exists = os.path.exists(os.path.join(self.save_path, folder_name, file_name))
            with open(os.path.join(self.save_path, folder_name, file_name), "a") as file:
                if not file_exists:
                    file.write(f"{'actor_' + str(i) for i in range(len(q_vals))}\n")
                file.write(f"{','.join(map(str, q_vals.detach().tolist()))}\n")

