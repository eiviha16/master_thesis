import numpy as np
import os
import yaml
from algorithms.misc.batch_buffer import Batch_VPG as Batch
import torch
from tqdm import tqdm


class VPG:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n.size
        self.obs_space_size = env.observation_space.shape[0]
        self.gamma = config['gamma']
        self.policy = Policy(self.obs_space_size, self.action_space_size, config['hidden_size'],
                             config['learning_rate'])
        self.batch = Batch(config['batch_size'])

        self.epochs = config['epochs']

        # self.test_random_seeds = [random.randint(1, 100000) for _ in range(100)]
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081,
                                  71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479,
                                  73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852,
                                  99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798,
                                  12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329,
                                  72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872,
                                  37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567,
                                  27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569,
                                  32088, 21418, 60590, 49736]

        self.test_seeds = np.random
        self.save = config['save']
        self.save_path = ''
        self.run_id = 'run_' + str(len([i for i in os.listdir(f"./results/{config['algorithm']}")]) + 1)
        self.make_run_dir(config['algorithm'])

        self.best_score = float('-inf')
        self.save_config(config)
        self.announce()

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def save_config(self, config):
        with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    def normalize_discounted_rewards(self):
        dr = np.array(self.batch.discounted_rewards)
        norm_dr = (dr - dr.mean()) / (dr.std() + 1e-10)
        self.batch.discounted_rewards = norm_dr

    def calculate_discounted_rewards(self):
        discounted_reward = 0
        for i in reversed(range(len(self.batch.actions))):
            discounted_reward = self.batch.rewards[i] + self.gamma * discounted_reward
            self.batch.discounted_rewards.insert(0, discounted_reward)

    def rollout(self):
        obs, _ = self.env.reset(seed=42)
        while True:
            action, log_prob = self.policy.get_action(obs)
            obs, reward, done, truncated, _ = self.env.step(action.detach().numpy())
            self.batch.save_experience(action, log_prob[action], obs, reward, done)
            if done or truncated:
                break

    def evaluate_actions(self):
        actions, log_probs = self.policy.get_action(self.batch.obs)
        return actions, log_probs

    def calculate_loss(self):
        return -(self.batch.action_log_prob * self.batch.discounted_rewards).mean()

    def train(self):
        actor_loss = self.calculate_loss()
        self.policy.actor_optim.zero_grad()
        actor_loss.backward()
        self.policy.actor_optim.step()

    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            self.rollout()
            self.calculate_discounted_rewards()
            self.normalize_discounted_rewards()
            self.batch.convert_to_tensor()

            self.train()
            self.test()

            self.batch.clear()

    def test(self):
        # remember to remove exploration when doing this
        episode_rewards = np.array([0 for _ in range(100)])
        for episode, seed in enumerate(self.test_random_seeds):
            obs, _ = self.env.reset(seed=seed)
            while True:
                action = self.policy.get_best_action(obs)
                obs, reward, done, truncated, _ = self.env.step(action.detach().numpy())
                episode_rewards[episode] += reward
                if done or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        self.save_results(mean, std)

        if mean > self.best_score:
            self.save_model('best_model')
            self.best_score = mean
            print(f'New best mean: {mean}!')
        self.save_model('last_model')

    def save_model(self, file_name):
        torch.save(self.policy, os.path.join(self.save_path, file_name))

    def save_results(self, mean, std):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.save_path, file_name))

        with open(os.path.join(self.save_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std\n")
            file.write(f"{mean},{std}\n")

    def make_run_dir(self, algorithm):
        base_dir = './results'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(os.path.join(base_dir, algorithm)):
            os.makedirs(os.path.join(base_dir, algorithm))
        if not os.path.exists(os.path.join(base_dir, algorithm, self.run_id)):
            os.makedirs(os.path.join(base_dir, algorithm, self.run_id))
        self.save_path = os.path.join(base_dir, algorithm, self.run_id)
