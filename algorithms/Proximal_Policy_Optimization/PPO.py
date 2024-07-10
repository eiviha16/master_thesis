import numpy as np
import os
import yaml
import random
from algorithms.misc.batch_buffer import Batch
import torch
import torch.nn.functional as F
from tqdm import tqdm

class PPO:
    def __init__(self, env, Policy, config):
        self.env = env
        self.action_space_size = env.action_space.n
        self.obs_space_size = env.observation_space.shape[0]

        self.policy = Policy(self.obs_space_size, self.action_space_size, config['hidden_size'], config['learning_rate'])
        self.batch = Batch(config['batch_size'])

        self.gamma = config['gamma']
        self.lam = config['lam']
        self.clip_range = config['clip_range']
        self.epochs = config['epochs']

        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081, 71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]

        self.save = config['save']
        self.save_path = ''
        self.config = config
        if self.save:
            self.run_id = 'run_' + str(len([i for i in os.listdir(f'../results/{config["env_name"]}/{config["algorithm"]}')]) + 1)
            self.make_run_dir()
        else:
            self.run_id = "undefined run"

        self.best_score = float('-inf')
        self.save_config(config)
        self.announce()
        self.cur_episode = 0
        self.scores = []

    def announce(self):
        print(f'{self.run_id} has been initialized!')

    def save_config(self, config):
        if self.save:

            with open(f'{self.save_path}/config.yaml', "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)

    def calculate_advantage(self):
        advantage = 0
        discounted_reward = 0
        next_value = 0

        for i in reversed(range(len(self.batch.actions))):
            if self.batch.trunc[i]:
                next_value = self.policy.critic(torch.tensor(self.batch.obs[i])).detach().numpy()
            dt = self.batch.rewards[i] + self.gamma * next_value * int(not self.batch.terminated[i]) - self.batch.values[i]
            advantage = dt + self.gamma * self.lam * advantage * int(not self.batch.terminated[i])
            next_value = self.batch.values[i]
            self.batch.advantages.insert(0, advantage)
            discounted_reward = self.batch.rewards[i] + self.gamma * discounted_reward
            self.batch.discounted_rewards.insert(0, discounted_reward)

    def normalize_advantages(self):
        advantages = np.array(self.batch.advantages)
        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.batch.advantages = norm_advantages

    def rollout(self):
        obs, _ = self.env.reset(seed=random.randint(1, 100))

        while True:
            action, value, log_prob = self.policy.get_action(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())
            self.batch.save_experience(action.detach(), log_prob.detach().numpy(), value.detach().numpy(), obs, reward, terminated, truncated)
            self.batch.next_value = self.policy.critic(torch.tensor(obs))
            if len(self.batch.obs) > self.config['n_steps']:
                self.batch.convert_to_numpy()
                self.calculate_advantage()
                self.normalize_advantages()
                self.train()
                self.batch.clear()

            if terminated or truncated:
                break



    def evaluate_actions(self, batch_idx):
        actions, values, log_probs = self.policy.evaluate_action(self.batch.sampled_obs[batch_idx : batch_idx + self.batch.batch_size], self.batch.sampled_actions[batch_idx : batch_idx + self.batch.batch_size])
        return actions, values, log_probs

    def calculate_actor_loss(self, log_prob, batch_idx):
        ratio = torch.exp(log_prob - torch.from_numpy(self.batch.sampled_action_log_prob[batch_idx : batch_idx + self.batch.batch_size]))
        actor_loss = torch.from_numpy(self.batch.sampled_advantages[batch_idx : batch_idx + self.batch.batch_size]).squeeze() * ratio
        clipped_actor_loss = torch.from_numpy(self.batch.sampled_advantages[batch_idx : batch_idx + self.batch.batch_size]).squeeze() * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        actor_loss = -(torch.min(actor_loss, clipped_actor_loss)).mean()
        return actor_loss

    def calculate_critic_loss(self, values, batch_idx):
        return F.mse_loss(torch.from_numpy(self.batch.sampled_advantages[batch_idx : batch_idx + self.batch.batch_size] + self.batch.sampled_values[batch_idx : batch_idx + self.batch.batch_size]).to(dtype=torch.float32), values)

    def train(self):
        self.batch.shuffle()
        for _ in range(self.epochs):
            for batch_idx in range(0, len(self.batch.sampled_obs), self.batch.batch_size):

                _, values, log_prob = self.evaluate_actions(batch_idx)
                actor_loss = self.calculate_actor_loss(log_prob, batch_idx)
                critic_loss = self.calculate_critic_loss(values, batch_idx)

                self.policy.actor_optim.zero_grad()
                actor_loss.backward()
                self.policy.actor_optim.step()

                self.policy.critic_optim.zero_grad()
                critic_loss.backward()
                self.policy.critic_optim.step()

    def learn(self, nr_of_episodes):
        for episode in tqdm(range(nr_of_episodes)):
            if episode % self.config['test_freq'] == 0:
                self.test()
            self.cur_episode = episode + 1
            self.rollout()


    def test(self):
        # remember to remove exploration when doing this
        episode_rewards = np.array([0 for _ in range(100)])
        with torch.no_grad():
            for episode, seed in enumerate(self.test_random_seeds):
                obs, _ = self.env.reset(seed=seed)
                while True:
                    action, probs = self.policy.get_best_action(obs)
                    obs, reward, terminated, truncated, _ = self.env.step(action.detach().numpy())
                    episode_rewards[episode] += reward
                    if terminated or truncated:
                        break

            mean = np.mean(episode_rewards)
            std = np.std(episode_rewards)
            self.save_results(mean, std)
            self.scores.append(mean)
            if mean > self.best_score:
                self.save_model('best_model')
                self.best_score = mean
                print(f'New best mean: {mean}!')
            self.save_model('last_model')

    def save_model(self, file_name):
        if self.save:
            torch.save(self.policy, os.path.join(self.save_path, file_name))

    def save_results(self, mean, std):
        if self.save:
            file_name = 'test_results.csv'
            file_exists = os.path.exists(os.path.join(self.save_path, file_name))

            with open(os.path.join(self.save_path, file_name), "a") as file:
                if not file_exists:
                    file.write("mean,std\n")
                file.write(f"{mean},{std}\n")
    def save_probs(self, probs):
        if self.save:

            folder_name = 'action_probabilities'
            file_name = f'{self.cur_episode}.csv'
            if not os.path.exists(os.path.join(self.save_path, folder_name)):
                os.makedirs(os.path.join(self.save_path, folder_name))
            file_exists = os.path.exists(os.path.join(self.save_path, folder_name, file_name))
            with open(os.path.join(self.save_path, folder_name, file_name), "a") as file:
                if not file_exists:
                    file.write(f"{'actor_' + str(i) for i in range(len(probs))}\n")
                #file.write(f"{q_vals[0]},{q_vals[1]}\n")
                file.write(f"{','.join(map(str, probs.detach().tolist()))}\n")


    def make_run_dir(self):
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
