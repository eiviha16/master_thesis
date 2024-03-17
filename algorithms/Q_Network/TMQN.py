import numpy as np
import torch
import os
import yaml
from tqdm import tqdm
import random

from algorithms.misc.replay_buffer import ReplayBuffer
#from algorithms.misc.plot_test_results import plot_test_results, feedback


class TMQN:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n
        self.obs_space_size = env.observation_space.shape[0]
        config['action_space_size'] = self.action_space_size
        config['obs_space_size'] = self.obs_space_size

        self.policy = Policy(config)

        self.gamma = config['gamma']  # discount factor
        self.exploration_prob = config['exploration_prob_init']
        self.exploration_prob_decay = config['exploration_prob_decay']

        self.epochs = config['epochs']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        #self.dynamic_memory = config['dynamic_memory']

        self.y_max = config['y_max']
        self.y_min = config['y_min']

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.test_freq = config['test_freq']
        self.nr_of_test_episodes = 100

        self.run_id = 'run_' + str(len([i for i in os.listdir(f'./results/{config["algorithm"]}')]) + 1)
        #self.test_random_seeds = [random.randint(1, 100000) for i in range(self.nr_of_test_episodes)]
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081, 71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]

        self.save = config['save']
        self.best_scores = {'mean': 0, 'std': float('inf')}
        self.cur_mean = 0
        self.config = config
        self.save_path = ''
        self.make_run_dir()
        self.save_config()
        self.announce()
        self.q_values = {f'q{i}': [] for i in range(self.action_space_size)}
        self.nr_actions = 0
        self.cur_episode = 0
        self.abs_errors = {}
        self.total_score = []

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
            q_vals = np.array([np.random.random() for _ in range(self.action_space_size)])
        else:
            q_vals = self.policy.predict(cur_obs)
            #self.q_values['q1'].append(q_vals[0])
            #self.q_values['q2'].append(q_vals[0])
        return np.argmax(q_vals), q_vals

    def temporal_difference(self, next_q_vals):
        return np.array(self.replay_buffer.sampled_rewards) + (
                1 - np.array(self.replay_buffer.sampled_dones)) * self.gamma * next_q_vals

    def update_exploration_prob(self):
        self.exploration_prob = self.exploration_prob * np.exp(-self.exploration_prob_decay)

    def get_q_val_and_obs_for_tm(self, target_q_vals):

        tm_inputs = [{'observations': [], 'target_q_vals': []} for _ in range(self.action_space_size)]
        #tm_1_input, tm_2_input = [{'observations': [], 'target_q_vals': []}, {'observations': [], 'target_q_vals': []}
        actions = self.replay_buffer.sampled_actions
        for index, action in enumerate(actions):
            tm_inputs[action]['observations'].append(self.replay_buffer.sampled_cur_obs[index])
            tm_inputs[action]['target_q_vals'].append(target_q_vals[index])
            """if action == 0:
                tm_1_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                tm_1_input['target_q_vals'].append(target_q_vals[index])

            elif action == 1:
                tm_2_input['observations'].append(self.replay_buffer.sampled_cur_obs[index])
                tm_2_input['target_q_vals'].append(target_q_vals[index])

            else:
                print('Error with get_q_val_for_action')
"""
        return tm_inputs #tm_1_input, tm_2_input

    def train(self):
        for epoch in range(self.epochs):
            self.replay_buffer.clear_cache()
            self.replay_buffer.sample()
            next_q_vals = self.policy.predict(np.array(self.replay_buffer.sampled_next_obs))  # next_obs?

            next_q_vals = np.max(next_q_vals, axis=1)

            # calculate target q vals
            target_q_vals = self.temporal_difference((next_q_vals))
            tm_1_input, tm_2_input = self.get_q_val_and_obs_for_tm(target_q_vals)
            tm_inputs = self.get_q_val_and_obs_for_tm(target_q_vals)
            abs_errors = self.policy.update(tm_inputs)

            for key in abs_errors:
                if key not in self.abs_errors:
                    self.abs_errors[key] = []
                for val in abs_errors[key]:
                    self.abs_errors[key].append(val)

        self.save_abs_errors()
        self.abs_errors = {}

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
            self.cur_episode = episode
            #actions = [0, 0]
            if self.test_freq:
                if episode % self.test_freq == 0:
                    self.test(nr_of_steps)
                    self.config['nr_of_episodes'] = episode + 1
                    self.config['nr_of_steps'] = nr_of_steps
                    self.save_config()
            cur_obs, _ = self.env.reset(seed=random.randint(1, 10000))
            episode_reward = 0

            while True:
                action, _ = self.get_next_action(cur_obs)
                #actions[action] += 1
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
        #plot_test_results(self.save_path, text={'title': 'TMQN'})

    def test(self, nr_of_steps):

        exploration_prob = self.exploration_prob
        self.exploration_prob = 0
        episode_rewards = np.array([0 for _ in range(self.nr_of_test_episodes)])

        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])  # episode)
            while True:
                action, q_vals_ = self.get_next_action(obs)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break
                if episode == 1:
                    self.save_q_vals(q_vals_)
        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.total_score.append(mean)
        self.cur_mean = mean
        self.save_results(mean, std, nr_of_steps)
        self.exploration_prob = exploration_prob
        if mean > self.best_scores['mean']:
            self.save_model(True)
            self.best_scores['mean'] = mean
            print(f'New best mean after {nr_of_steps} steps: {mean}!')
        self.save_model(False)
        #self.save_q_vals(nr_of_steps)

    def save_model(self, best_model):
        if best_model:
            #self.policy.tm1.save_state()
            #self.policy.tm2.save_state()
            #tms = [self.policy.tm1, self.policy.tm2]
            tms = self.policy.tms
            tms_save = []
            for tm in range(len(tms)):
                ta_state, clause_sign, clause_output, feedback_to_clauses = tms[tm].get_params()
                ta_state_save = np.zeros((len(ta_state), len(ta_state[0]), len(ta_state[0][0])), dtype=np.int32)
                clause_sign_save = np.zeros((len(clause_sign)), dtype=np.int32)
                clause_output_save = np.zeros((len(clause_output)), dtype=np.int32)
                feedback_to_clauses_save = np.zeros((len(feedback_to_clauses)), dtype=np.int32)

                for i in range(len(ta_state)):
                    for j in range(len(ta_state[i])):
                        for k in range(len(ta_state[i][j])):
                            ta_state_save[i][j][k] = int(ta_state[i][j][k])
                    clause_sign_save[i] = int(clause_sign[i])
                    clause_output_save[i] = int(clause_output[i])
                    feedback_to_clauses_save[i] = int(feedback_to_clauses[i])
                tms_save.append({'ta_state': ta_state_save, 'clause_sign': clause_sign_save, 'clause_output': clause_output_save, 'feedback_to_clauses': feedback_to_clauses_save})
            torch.save(tms_save, os.path.join(self.save_path, 'best'))

        else:
            pass

    def save_results(self, mean, std, nr_of_steps):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std, steps\n")
            file.write(f"{mean},{std},{nr_of_steps}\n")

    def save_q_vals(self, q_vals):
        folder_name = 'q_values'
        file_name = f'{self.cur_episode}.csv'
        if not os.path.exists(os.path.join(self.save_path, folder_name)):
            os.makedirs(os.path.join(self.save_path, folder_name))
        file_exists = os.path.exists(os.path.join(self.save_path, folder_name, file_name))

        with open(os.path.join(self.save_path, folder_name, file_name), "a") as file:
            if not file_exists:
                file.write(f"{'actor_' + str(i) for i in range(len(q_vals))}\n")
            #file.write(f"{q_vals[0][0]}, {q_vals[0][1]}\n")
            file.write(f"{','.join(map(str, q_vals))}\n")

    def save_abs_errors(self):
        for key in self.abs_errors:
            self.abs_errors[key] = np.array(self.abs_errors[key])
        folder_name = 'absolute_errors.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, folder_name))

        with open(os.path.join(self.save_path, folder_name), "a") as file:
            if not file_exists:
                file.write('actor1_mean,actor1_std,actor2_mean,actor2_std\n')
            file.write(f"{np.mean(self.abs_errors['actor1'])},{np.std(self.abs_errors['actor1'])},{np.mean(self.abs_errors['actor2'])},{np.std(self.abs_errors['actor2'])}\n")
