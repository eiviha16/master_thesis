import gymnasium as gym
from stable_baselines3 import PPO
import os


def save(obs, file_path):
    obs_str = [str(elem) for elem in obs.tolist()]
    obs = ','.join(obs_str)
    with open(file_path, "a") as file:
        file.write(obs + '\n')


env = gym.make("Acrobot-v1", render_mode="human")

file_path = "baseline/ppo_acrobot-3"
model = PPO.load(file_path)
file_name = 'acrobot_obs_data.txt'

for i in range(10):
    obs, info = env.reset()
    rewards = 0
    while True:

        action, _ = model.predict(obs)
        save(obs, file_name)
        obs, reward, done, trunc, info = env.step(action)
        rewards += reward
        if done or trunc:
            break
    print(rewards)
