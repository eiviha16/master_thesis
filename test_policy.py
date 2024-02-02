import numpy as np
import torch
import random

random.seed(42)
np.random.seed(1)
torch.manual_seed(42)

import gymnasium as gym

def test_policy(save_file, policy):
    #seeds = [np.random.randint(10000000, 100000000) for i in range(100)]
    seeds = [75682867, 66755036, 66882282, 31081788, 23315092, 45788921, 36735830, 41632483, 86737383, 98358551, 98409749, 14981505, 23953367, 95652971, 14521373, 13344769, 86893497, 40349564, 52860080, 87751354, 72250665, 76209791, 56792155, 31498555, 70221198, 89757501, 26861870, 62286002, 42049003, 71136438, 72194931, 75285250, 79537252, 69248434, 71306900, 65831368, 20959014, 66972561, 58900483, 30193880, 13385357, 54738553, 78574553, 26845364, 62157313, 21392366, 79778859, 98883975, 23479854, 83506850, 23187277, 78979792, 47709731, 36939239, 33027075, 91200125, 96191493, 97796277, 76401385, 32335235, 10271836, 13584702, 62631083, 18585377, 96544585, 95157821, 97655395, 27824013, 46601694, 77105583, 70304654, 44119117, 46433622, 99482491, 70031992, 65105831, 13366612, 68743503, 46258670, 79739572, 65848857, 67055419, 59160571, 78394024, 74525468, 99966606, 48727468, 61757120, 77157848, 87897542, 83665032, 95390464, 58170987, 62562567, 33717335, 70472382, 22719242, 58715250, 93139152, 54054178]

    env = gym.make("CartPole-v1")

    episode_rewards = np.array([0 for _ in range(100)])
    actions = 0
    for episode in range(100):
        obs, _ = env.reset(seed=seeds[episode])  # episode)
        #print(episode)
        while True:
            try:
                action = np.argmax(policy.predict(obs))
                obs, reward, done, truncated, _ = env.step(action)
            except:
                action = torch.argmax(policy(torch.tensor(obs)))
                obs, reward, done, truncated, _ = env.step(action.detach().numpy())
            episode_rewards[episode] += reward
            actions += 1

            if done or truncated:
                break

    mean = np.mean(episode_rewards)
    std = np.std(episode_rewards)
    import os
    print(f'Mean reward: {mean}')
    print(f'Mean std: {std}')
    print(f'Actions: {actions}')

    with open(save_file, 'w') as file:
        file.write(episode_rewards)

if __name__ == '__main__':
    file = './results/PPO/run_38/best_model'
    model = torch.load(file)
    test_policy(model.actor)