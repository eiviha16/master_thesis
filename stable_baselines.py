import torch
import numpy as np
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import gymnasium as gym
from stable_baselines3 import PPO, DQN
import os
from tqdm import tqdm

class PPO_c():
    def __init__(self, config):
        self.nr_of_test_episodes = 100
        self.test_random_seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081,
                                  71483,
                                  11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479, 73564,
                                  26063, 93851,
                                  85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852, 99459, 20927, 91507,
                                  55393,
                                  44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798, 12677, 47053, 45083, 79132,
                                  34672,
                                  5696, 95648, 60218, 70285, 16362, 49616, 10329, 72358, 38428, 82398, 81071, 47401,
                                  75675,
                                  25204, 92350, 9117, 6007, 86674, 29872, 37931, 10459, 30513, 13239, 49824, 36435,
                                  59430,
                                  83321, 47820, 21320, 48521, 46567, 27461, 87842, 34994, 91989, 89594, 84940, 9359,
                                  79841,
                                  83228, 22432, 70011, 95569, 32088, 21418, 60590, 49736]

        self.best_score = - float("inf")
        self.config = config
        #self.n_timesteps = 100
        self.n_timesteps = 100
        self.file_path = f'./results/{self.config["env_name"]}/{self.config["algorithm"]}/{self.config["run"]}'

        # Parallel environments
        if self.config["env_name"] == "cartpole":
            self.env = gym.make("CartPole-v1")
        elif self.config["env_name"] == "acrobot":
            self.env = gym.make("Acrobot-v1")
        self.timesteps = 0

        self.model = PPO("MlpPolicy", self.env, seed=42, verbose=0)#, n_steps=16)
       # self.model = DQN("MlpPolicy", self.env, seed=42, verbose=0)#, n_steps=16)

    def learn(self, intervals):
        for i in tqdm(range(intervals)):
            self.test_environment()
            self.model.learn(total_timesteps=self.n_timesteps)
            self.timesteps += self.n_timesteps

    def test_environment(self):

        episode_rewards = np.array([0 for _ in range(self.nr_of_test_episodes)])

        for episode in range(self.nr_of_test_episodes):
            obs, _ = self.env.reset(seed=self.test_random_seeds[episode])  # episode)
            while True:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.env.step(action)
                episode_rewards[episode] += reward
                if done or truncated:
                    break

        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)

        self.save_results(mean, std)

        if mean > self.best_score:
            self.best_score = mean
            print(f'New best mean: {mean}!')
            self.model.save(f"{self.file_path}/best")
            # self.save_q_vals(nr_of_steps)

    def save_results(self, mean, std):
        file_name = 'test_results.csv'
        file_exists = os.path.exists(os.path.join(self.file_path, file_name))

        with open(os.path.join(self.file_path, file_name), "a") as file:
            if not file_exists:
                file.write("mean,std,steps\n")
            file.write(f"{mean},{std},{self.timesteps}\n")


if __name__ == "__main__":
    #config = {"env_name": "acrobot", "algorithm": "PPO", "run": "run_3"}
    config = {"env_name": "cartpole", "algorithm": "PPO", "run": "run_1f"}
    ppo = PPO_c(config)
    ppo.learn(1000)

    import test_policy
    model = ppo.model.load(f"{ppo.file_path}/best")
    test_policy.test_policy(ppo.file_path, model, ppo.config["env_name"], sb=True)
