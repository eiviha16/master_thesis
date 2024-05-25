from matplotlib import pyplot as plt
import csv
import numpy as np


def plot_many(names, data, title, ratio):
    plt.figure(figsize=(7, 5.5))
    for key in data:
        #x = np.arange(0, int((len(data[key]['mean'])) * ratio), step=ratio)
        x = np.arange(0, int(len(data[key]['mean']) * 10), 10)
        #plt.plot(np.array(data[key]['steps']), data[key]['mean'], label=names[key])
        plt.plot(x, data[key]['mean'], label=names[key])
        """plt.fill_between(np.array(data[key]['steps']), np.array(data[key]['mean']) - np.array(data[key]['std']),
                         np.array(data[key]['mean']) + np.array(data[key]['std']),
                         alpha=0.10)"""
        plt.fill_between(x, np.array(data[key]['mean']) - np.array(data[key]['std']),
                         np.array(data[key]['mean']) + np.array(data[key]['std']),
                         alpha=0.10)

    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Mean rewards')
    plt.xlabel(f'Episodes')
    plt.title(f'{title}')
    plt.legend()
    # plt.savefig(f'rewards_comparison.png')
    plt.savefig("plot", format='svg')
    plt.show()


def get_csv_performance(file_path, clipped=False):
    data = {'mean': [], 'std': [], 'steps': []}
    if not clipped:
        fp = f'{file_path}/test_results.csv'
    else:
        fp = f'{file_path}/test_results_clipped.csv'
    with open(fp, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'mean':
                data['mean'].append(float(row[0]))
                data['std'].append(float(row[1]))
                data['steps'].append(float(row[2]))
        data['mean'] = data['mean'][:50000]
        data['std'] = data['std'][:50000]
        data['steps'] = data['steps'][:50000]
    return data


def plot_test_results(file_path, text):
    data = get_csv_performance(file_path)
    plot(data, text, file_path)


def prune(data, new_size):
    for key in data:
        for m in data[key]:
            if new_size != -1:
                ratio = int(len(data[key][m]) / new_size)
            else:
                ratio = 1
            new_data = []
            for i in range(len(data[key][m])):
                if i % ratio == 0 or new_size == -1:
                    new_data.append(data[key][m][i])
            data[key][m] = new_data
    return data, ratio


def prune_2(data, new_size):
    intervals = {}
    for key in data:
        timesteps = data[key]['steps'][-1]
        intervals[key] = timesteps / new_size
        for m in data[key]:
            new_data = []
            _steps = intervals[key]
            plot_points = 0
            for i in range(len(data[key][m])):
                if data[key]['steps'][i] > _steps or i == 0:
                    new_data.append(data[key][m][i])
                    plot_points += 1
                    _steps += intervals[key]
                if plot_points == new_size:
                    break
            data[key][m] = new_data
    ratio = 1
    return data, ratio


def plot_many_rewards(environment, algorithms, new_size):
    data = {}
    names = {}
    for algorithm in algorithms:
        # data[algorithm] = get_csv_performance(f'../../results/{algorithm}/{algorithms[algorithm]}')
        # data[algorithm] = get_csv_performance(f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
        if environment == 'Cartpole':
            if algorithm == "DQN" and algorithm == "TAC random":
                # data[algorithm] = get_csv_performance(f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}', True)
                data[algorithm] = get_csv_performance(
                    f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
            else:
                data[algorithm] = get_csv_performance(
                    f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
        elif environment == "Acrobot":
            data[algorithm] = get_csv_performance(
                f'../final_results/acrobot/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
        names[algorithm] = algorithms[algorithm]["name"]
    # title = f'{environment} - Q-Tsetlin-Machine'
    # title = f'{environment} - n-step Q-Tsetlin-Machine'
    # title = f'{environment} - Baselines'
    # title = 'Cartpole - Actor-Critics'
    # title = f'{environment} - Actor-Critics'
    # title = f'{environment} - Double Q-Tsetlin-Machine - Type b update'
    # title = f'{environment} - n-step Double Q-Tsetlin-Machine - Type b update'
    # title = f'{environment} - n-step Q-Tsetlin-Machine'
    # title = f'{environment} - Deep Q-Network'
    # title = f'{environment} - n-step Deep Q-Network'
    title = f'{environment} '
    # title = f'{environment} - Tsetlin Proximal Policy Optimization'
    # title = f'{environment} - Proximal Policy Optimization'
    if new_size != -1:
        data, ratio = prune(data, new_size)
    else:
        ratio = 1
    plot_many(names, data, title, ratio)


if __name__ == "__main__":
    ################## cartpole ##############################
    ################## cartpole ##############################
    ################## cartpole ##############################
    ################## cartpole ##############################
    """algorithms = {
        'DQN': {'folder': 'DQN', 'run': 'run_91', 'name': 'DQN'},
        # 'PPO': {'folder': 'PPO', 'run': 'run_3_final', 'name': 'PPO'},
        # 'TAC random': {'folder': 'TAC_random', 'run': 'run_2', 'name': 'TAC random'}
    }

    algorithms = {
        #    'TPPO': {'folder': 'TPPO', 'run': 'run_11', 'name': 'TPPO'},
        'TAC \n Update type a': {'folder': 'TAC_a', 'run': 'run_32', 'name': 'Tsetlin Actor-Critic - Type a update'},
        #    'TAC \n Update type b': {'folder': 'TAC_b', 'run': 'run_9', 'name': 'Tsetlin Actor-Critic - Type b update'},
    }"""
    """algorithms = {
        'QTM': {'folder': 'QTM', 'run': 'run_2', 'name': 'QTM'},
        'DQTM \n Update type a': {'folder': 'Double_QTM_a', 'run': 'run_4', 'name': 'DQTM - Type a update'},
        'DQTM \n Update type b': {'folder': 'Double_QTM_b', 'run': 'run_2', 'name': 'DQTM - Type b update'},
    }"""

    """algorithms = {
        'n-step QTM': {'folder': 'n_step_QTM', 'run': 'run_2', 'name': 'n-step QTM'},
        'n-step DQTM \n Update type a': {'folder': 'n_step_Double_QTM_a', 'run': 'run_2', 'name': 'n-step DQTM - Type a update'},
        'n-step DQTM \n Update type b': {'folder': 'n_step_Double_QTM_b', 'run': 'run_7', 'name': 'n-step DQTM - Type b update'},
    }"""
    #########¤ Acrobot ##############
    #########¤ Acrobot ##############
    #########¤ Acrobot ##############
    #########¤ Acrobot ##############
    #########¤ Acrobot ##############
    """algorithms = {
        'n-step DQN': {'folder': 'n_step_DQN', 'run': 'run_66', 'name': 'n-step DQN'},#20
        'PPO': {'folder': 'PPO', 'run': 'run_2', 'name': 'PPO'},
        'TAC random': {'folder': 'TAC_random', 'run': 'run_12', 'name': 'TAC random'}
    }"""
    """algorithms = {
        'TPPO': {'folder': 'TPPO', 'run': 'run_6', 'name': 'TPPO'},
        'TAC \n Update type a': {'folder': 'TAC_a', 'run': 'run_5', 'name': 'Tsetlin Actor-Critic - Type a update'},
        'TAC \n Update type b': {'folder': 'TAC_b', 'run': 'run_13', 'name': 'Tsetlin Actor-Critic - Type b update'},
    }"""

    """algorithms = {
        'QTM': {'folder': 'QTM', 'run': 'run_8', 'name': 'QTM'},
        'DQTM \n Update type a': {'folder': 'Double_QTM_a', 'run': 'run_14', 'name': 'DQTM - Type a update'},
        'DQTM \n Update type b': {'folder': 'Double_QTM_b', 'run': 'run_1b', 'name': 'DQTM - Type b update'},
    }"""
    """algorithms = {
        'n-step QTM': {'folder': 'n_step_QTM', 'run': 'run_1', 'name': 'n-step QTM'},
        'n-step DQTM \n Update type a': {'folder': 'n_step_Double_QTM_a', 'run': 'run_4', 'name': 'n-step DQTM - Type a update'},
        'n-step DQTM \n Update type b': {'folder': 'n_step_Double_QTM_b', 'run': 'run_4', 'name': 'n-step DQTM - Type b update'},
    }"""
    """algorithms = {
        'n-step QTM': {'folder': 'n_step_QTM', 'run': 'run_1', 'name': 'n-step QTM'},
        'n-step DQTM \n Update type a': {'folder': 'n_step_Double_QTM_a', 'run': 'run_3', 'name': 'n-step DQTM - Type a update'},
        'n-step DQTM \n Update type b': {'folder': 'n_step_Double_QTM_b', 'run': 'run_4', 'name': 'n-step DQTM - Type b update'},
    }"""
    """algorithms = {
        #'DQN': {'folder': 'DQN', 'run': 'run_10', 'name': 'DQN'},
        'n-step DQTM \n Update type b': {'folder': 'n_step_Double_QTM_b', 'run': 'run_4',
                                         'name': 'n-step DQTM - Type b update'},

    }"""

    """algorithms = {
        'n-step DQN': {'folder': 'n_step_DQN', 'run': 'run_66', 'name': 'n-step DQN'},#20
        #'PPO': {'folder': 'PPO', 'run': 'run_2', 'name': 'PPO'},
        #'TAC random': {'folder': 'TAC_random', 'run': 'run_11', 'name': 'TAC random'}
    }"""

    algorithms = {
        'TPPO': {'folder': 'TPPO', 'run': 'run_64', 'name': 'Tsetlin Proximal Policy Optimization'},
        #'TAAC': {'folder': 'TAAC', 'run': 'run_1', 'name': 'Tsetlin Advantage Actor-Critic'},
        #'TAC \n Update type a': {'folder': 'TAC_a', 'run': 'run_34', 'name': 'Tsetlin Actor-Critic - Type a update'},
        #    'TAC \n Update type b': {'folder': 'TAC_b', 'run': 'run_9', 'name': 'Tsetlin Actor-Critic - Type b update'},
    }
    plot_many_rewards('Cartpole', algorithms, new_size=-1)
# 'n_step_Double_TMQN': 'run_34' 498.22 - 11.22
# 'n_step_Double_TMQN': 'run_35' 500.0 - 0.0
