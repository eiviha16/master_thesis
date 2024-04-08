import numpy as np
import torch
import random

random.seed(42)
np.random.seed(1)
torch.manual_seed(42)
import os
import gymnasium as gym


def save_action_vals(fp, episode, action_vals):
    folder_name = 'final_test_action_vals'
    file_name = f'{episode}.csv'
    if not os.path.exists(os.path.join(fp, folder_name)):
        os.makedirs(os.path.join(fp, folder_name))

    file_exists = os.path.exists(os.path.join(fp, folder_name, file_name))
    with open(os.path.join(fp, folder_name, file_name), "a") as file:
        if not file_exists:
            file.write(f"{'actor_' + str(i) for i in range(len(action_vals))}\n")
        file.write(f"{','.join(map(str, action_vals.detach().tolist()))}\n")
def save_vals(fp, episode, action_vals):
    folder_name = 'final_test_action_vals'
    file_name = f'{episode}.csv'
    if not os.path.exists(os.path.join(fp, folder_name)):
        os.makedirs(os.path.join(fp, folder_name))

    file_exists = os.path.exists(os.path.join(fp, folder_name, file_name))
    with open(os.path.join(fp, folder_name, file_name), "a") as file:
        if not file_exists:
            file.write("actor_1,actor_2\n")
        file.write(f"{action_vals[0]}, {action_vals[1]}\n")
def test_policy(save_file, policy, env_name, sb=False):
    #seeds = [np.random.randint(10000000, 100000000) for i in range(100)]
    seeds = [75682867, 66755036, 66882282, 31081788, 23315092, 45788921, 36735830, 41632483, 86737383, 98358551, 98409749, 14981505, 23953367, 95652971, 14521373, 13344769, 86893497, 40349564, 52860080, 87751354, 72250665, 76209791, 56792155, 31498555, 70221198, 89757501, 26861870, 62286002, 42049003, 71136438, 72194931, 75285250, 79537252, 69248434, 71306900, 65831368, 20959014, 66972561, 58900483, 30193880, 13385357, 54738553, 78574553, 26845364, 62157313, 21392366, 79778859, 98883975, 23479854, 83506850, 23187277, 78979792, 47709731, 36939239, 33027075, 91200125, 96191493, 97796277, 76401385, 32335235, 10271836, 13584702, 62631083, 18585377, 96544585, 95157821, 97655395, 27824013, 46601694, 77105583, 70304654, 44119117, 46433622, 99482491, 70031992, 65105831, 13366612, 68743503, 46258670, 79739572, 65848857, 67055419, 59160571, 78394024, 74525468, 99966606, 48727468, 61757120, 77157848, 87897542, 83665032, 95390464, 58170987, 62562567, 33717335, 70472382, 22719242, 58715250, 93139152, 54054178]
    """seeds = [83811, 14593, 3279, 97197, 36049, 32099, 29257, 18290, 96531, 13435, 88697, 97081,
                              71483, 11396, 77398, 55303, 4166, 3906, 12281, 28658, 30496, 66238, 78908, 3479,
                              73564, 26063, 93851, 85182, 91925, 71427, 54988, 28894, 58879, 77237, 36464, 852,
                              99459, 20927, 91507, 55393, 44598, 36422, 20380, 28222, 44119, 13397, 12157, 49798,
                              12677, 47053, 45083, 79132, 34672, 5696, 95648, 60218, 70285, 16362, 49616, 10329,
                              72358, 38428, 82398, 81071, 47401, 75675, 25204, 92350, 9117, 6007, 86674, 29872,
                              37931, 10459, 30513, 13239, 49824, 36435, 59430, 83321, 47820, 21320, 48521, 46567,
                              27461, 87842, 34994, 91989, 89594, 84940, 9359, 79841, 83228, 22432, 70011, 95569,
                              32088, 21418, 60590, 49736]
    """
    if env_name == 'cartpole':
        env = gym.make("CartPole-v1")
    else:
        env = gym.make("Acrobot-v1")

    episode_rewards = np.array([0 for _ in range(100)])
    actions = 0
    action_vals = []
    for episode in range(100):
        obs, _ = env.reset(seed=seeds[episode])  # episode)
        #print(episode)
        while True:
            if sb:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)

            else:
                try:
                    action_val = policy.predict(obs)
                    action = np.argmax(action_val[0])
                    obs, reward, done, truncated, _ = env.step(action)
                    #save_vals(save_file, episode, action_val[0])

                except:
                    action_val = policy.predict(obs)#torch.tensor(obs))
                    action = np.argmax(action_val.detach().numpy())
                    obs, reward, done, truncated, _ = env.step(action)
                    #save_action_vals(save_file, episode, action_val)

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
    if not os.path.exists(save_file):
        os.makedirs(save_file)

    with open(os.path.join(save_file, 'final_test_results'), 'w') as file:
        for reward in episode_rewards:
            file.write(str(reward) + "\n")

if __name__ == '__main__':
    #file = './results/DQN/run_50/best_model'
    file = './algorithms/final_results/cartpole/DQN/run_10/best_model'
    #file = './results/PPO/run_117/best_model'
    model = torch.load(file)
    test_policy('./algorithms/final_results/cartpole/DQN/run_10/', model, "cartpole")