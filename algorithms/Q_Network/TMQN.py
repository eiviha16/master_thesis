import numpy as np
import torch
import os
import yaml
from tqdm import tqdm
import random

from algorithms.misc.replay_buffer import ReplayBuffer
from algorithms.misc.plot_test_results import plot_test_results, feedback


class TMQN:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]
        self.policy = Policy(config)

        self.gamma = config['gamma']  # discount factor
        self.exploration_prob = config['exploration_prob_init']
        self.exploration_prob_decay = config['exploration_prob_decay']

        self.epochs = config['epochs']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.dynamic_memory = config['dynamic_memory']

        self.y_max = config['y_max']
        self.y_min = config['y_min']

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 100

        self.run_id = 'run_' + str(len([i for i in os.listdir(f'./results/{config["algorithm"]}')]) + 1)
        self.test_random_seeds = [random.randint(1, 100000) for i in range(self.nr_of_test_episodes)]
        self.save = config['save']
        self.best_scores = {'mean': 0, 'std': float('inf')}
        self.cur_mean = 0
        self.config = config
        self.save_path = ''
        self.make_run_dir()
        self.save_config()
        self.announce()
        self.q_values = {'q1': [], 'q2': []}
        self.nr_actions = 0

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def make_run_dir(self):
        base_dir = './results'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(os.path.join(base_dir, self.config["algorithm"])):
            os.makedirs(os.path.join(base_dir, self.config["algorithm"]))
        if not os.path.exists(os.path.join(base_dir, self.config["algorithm"], self.run_id)):
            os.makedirs(os.path.join(base_dir, self.config["algorithm"], self.run_id))
        self.save_path = os.path.join(base_dir, self.config["algorithm"], self.run_id)

    def save_config(self):
        with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False)

    def get_next_action(self, cur_obs):
        if np.random.random() < self.exploration_prob:
            q_vals = np.array([np.random.random() for _ in range(self.action_space_size + 1)])
        else:
            q_vals = self.policy.predict(cur_obs)
            self.q_values['q1'].append(q_vals[0])
            self.q_values['q2'].append(q_vals[0])
        return np.argmax(q_vals)

    def temporal_difference(self, next_q_vals):
        return np.array(self.replay_buffer.sampled_rewards) + (
                1 - np.array(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals

    def update_exploration_prob(self):
        self.exploration_prob = self.exploration_prob * np.exp(-self.exploration_prob_decay)

    def get_q_val_and_obs_for_tm(self, target_q_vals):

        tm_1_input, tm_2_input = {'observations': [], 'target_q_vals': []}, {'observations': [], 'target_q_vals': []}
        actions = self.replay_buffer.sampled_actions
        for index, action in enumerate(actions):
            if action == 0:
                tm_1_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                tm_1_input['target_q_vals'].append(target_q_vals[index])

            elif action == 1:
                tm_2_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                tm_2_input['target_q_vals'].append(target_q_vals[index])

            else:
                print('Error with get_q_val_for_action')

        return tm_1_input, tm_2_input

    def train(self):
        for epoch in range(self.epochs):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample()
            next_q_vals = self.policy.predict(np.array(self.replay_buffer.sampled_next_obs))  # next_obs?

            next_q_vals = np.max(next_q_vals, axis=1)

            # calculate target q vals
            target_q_vals = self.temporal_difference((next_q_vals))
            tm_1_input, tm_2_input = self.get_q_val_and_obs_for_tm(target_q_vals)
            self.policy.update(tm_1_input, tm_2_input)

    def save_feedback_data(self, feedback_1, feedback_2, nr_of_steps):
        file_name = 'feedback.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("1_typeI, 1_typeII, 2_typeI, 2_typeII, steps\n")
            file.write(f"{feedback_1[0]},{feedback_1[1]},{feedback_2[0]},{feedback_2[1]},{nr_of_steps}\n")

    def learn(self, nr_of_episodes):
        nr_of_steps = 0
        for episode in tqdm(range(nr_of_episodes)):
            actions = [0, 0]
            if self.test_freq:
                if episode % self.test_freq == 0:
                    self.test(nr_of_steps)
                    self.config['nr_of_episodes'] = episode + 1
                    self.config['nr_of_steps'] = nr_of_steps
                    self.save_config()
                    if self.dynamic_memory:
                        # if self.has_reached_threshold:
                        intervals = 5
                        max_score = 500

                        new_memory_size = max(self.dynamic_memory_min_size, int(intervals * np.ceil(
                            self.dynamic_memory_max_size / intervals * intervals * self.cur_mean / max_score / intervals)))

                        self.policy.tm1.update_memory_size(new_memory_size)
                        self.policy.tm2.update_memory_size(new_memory_size)

            cur_obs, _ = self.env.reset(seed=random.randint(1, 10000))
            episode_reward = 0

            while True:
                action = self.get_next_action(cur_obs)
                actions[action] += 1
                next_obs, reward, done, truncated, _ = self.env.step(action)

                # might want to not have truncated in my replay buffer
                self.replay_buffer.save_experience(action, cur_obs, next_obs, reward, int(done), nr_of_steps)
                episode_reward += reward
                cur_obs = next_obs
                nr_of_steps += 1

                if done or truncated:
                    break
            if nr_of_steps >= self.batch_size:
                self.train()
                self.save_actions(actions, nr_of_steps)
            self.update_exploration_prob()
        plot_test_results(self.save_path, text={'title': 'TMQN'})

    def test(self, nr_of_steps):
        self.q_vals = [0, 0]
        self.nr_actions = 0
        exploration_prob = self.exploration_prob
        self.exploration_prob = 0
        episode_rewards = np.array([0 for _ in range(self.nr_of_test_episodes)])

        for episode in range(self.nr_of_test_episodes):
            self.q_values['q1'] = []
            self.q_values['q2'] = []
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])  # episode)
            while True:
                action = self.get_next_action(obs)
                self.nr_actions += 1
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.cur_mean = mean
        self.save_results(mean, std, nr_of_steps)
        self.exploration_prob = exploration_prob
        if mean > self.best_scores['mean']:
            self.save_model(True)
            self.best_scores['mean'] = mean
            print(f'New best mean after {nr_of_steps} steps: {mean}!')
        self.save_model(False)
        self.save_q_vals(nr_of_steps)

    def save_model(self, best_model):
        if best_model:
            self.policy.tm1.save_state()
            self.policy.tm2.save_state()
        else:
            pass

    def save_results(self, mean, std, nr_of_steps):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std, steps\n")
            file.write(f"{mean},{std},{nr_of_steps}\n")

    def save_actions(self, actions, nr_of_steps):
        file_name = 'actions.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("tm1,tm2,steps\n")
            file.write(f"{actions[0]},{actions[1]},{nr_of_steps}\n")

    def save_q_vals(self, nr_of_steps):
        folder_name = 'q_values'
        path = os.path.join(self.save_path, folder_name)
        formatted_q_vals = [f"{q1},{q1}\n" for q1, q2 in zip(self.q_values['q1'], self.q_values['q2'])]
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, str(nr_of_steps)), "a") as file:
            file.write('q1,q2\n')
            file.writelines(formatted_q_vals)
