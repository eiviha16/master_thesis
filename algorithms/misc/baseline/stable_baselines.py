import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Acrobot-v1")

model = PPO("MlpPolicy", env, verbose=1)

num_timesteps = 50_000
print('Running!')
model.learn(total_timesteps=num_timesteps, log_interval=5)

#mean_reward = evaluate_policy(model, env)
#print(f"Mean reward: {mean_reward}")

model.save("algorithms/misc/baseline/ppo_acrobot-3")

