import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Acrobot-v1", render_mode="human")

file_path = "ppo_acrobot-3.zip"
#model = PPO.load(file_path)


for i in range(10):
    obs, info = env.reset()
    rewards = 0
    while True:
        #action, _ = model.predict(obs)
        action = env.action_space.sample()
        obs, reward, done, info, trunc = env.step(action)
        rewards += reward
        if done or trunc:
            break
