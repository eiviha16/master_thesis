import seaborn as sns
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

"""
def plot_many(names, data, title, ratio):
    for key in data:
        x = np.arange(1, int((len(data[key]['mean'])) * ratio), step=ratio)
        plt.plot(x, data[key]['mean'], label=names[key])
        plt.fill_between(x, np.array(data[key]['mean']) - np.array(data[key]['std']),
                             np.array(data[key]['mean']) + np.array(data[key]['std']),
                             alpha=0.10)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Mean rewards')
    plt.xlabel(f'Episodes')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'plots/rewards_comparison.png')
    plt.show()
"""


def plot_violin(data, title, ratio):
    means = {key: values['mean'] for key, values in data.items()}
    max_length = max(len(v) for v in means.values())
    means = {k: v + [np.nan] * (max_length - len(v)) for k, v in means.items()}
    # Convert to DataFrame for Seaborn
    df = pd.DataFrame(means)
    """mapping = {"PPO": "Baselines",
               "n-step DQM": "Baselines",
               "TAC random": "Baselines",

               "QTM": "QTM",
               "DQTM \n Update type a": "QTM",
               "DQTM \n Update type b": "QTM",

               "n-step QTM": "n-step QTM",
               "n-step DQTM \n Update type a": "n-step QTM",
               "n-step DQTM \n Update type b": "n-step QTM",

               "TPPO": "Actor-Critics",
               "TAC \n Update type b": "Actor-Critics",
               "TAC \n Update type a": "Actor-Critics",
               }"""
    df_melted = df.melt(var_name='Algorithm', value_name='Mean')
    plt.figure(figsize=(7, 5.5))
    # df_melted['category'] = df_melted['Algorithm'].map(mapping)
    # plt.figure(figsize=(16, 5.5))
    # palette = sns.color_palette("Paired", 3)

    # df_melted['color'] = df_melted['Algorithm'].map(color_dict)

    sns.violinplot(x='Algorithm', y="Mean", data=df_melted)  # , palette=color_dict)
    # sns.violinplot(x='category', y="Mean", hue='Algorithm', data=df_melted)
    plt.title(title)
    # plt.axhline(y=-100, color='r', linestyle='--')
    # plt.axhline(y=475, color='r', linestyle='--')

    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel('Mean rewards')
    plt.xlabel("Algorithm")
    plt.savefig("plot", format='svg')

    plt.show()


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


def get_csv_performance(file_path, clipped=False):
    data = {'mean': [], 'std': []}
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
                # data['timesteps'].append(float(row[2]))
        data['mean'] = data['mean'][:10000]
        data['std'] = data['std'][:10000]
    return data


def get_csv_performance_2(file_path):
    data = {'mean': []}
    try:
        with open(f'{file_path}/final_test_results/final_test_results', 'r') as file:
            lines = file.readlines()
            for line in lines:
                score = line.split()
                score = int(score[0])
                data['mean'].append(score)
    except:
        with open(f'{file_path}/final_test_results', 'r') as file:
            lines = file.readlines()
            for line in lines:
                score = line.split()
                score = int(score[0])
                data['mean'].append(score)
    return data


def plot_many(environment, algorithms, new_size):
    data = {}
    names = []
    for algorithm in algorithms:
        if environment == 'Cartpole':
            if algorithm == "DQN" or algorithm == "TAC random":
                data[algorithm] = get_csv_performance(
                    f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
                # data[algorithm] = get_csv_performance(f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
            else:
                data[algorithm] = get_csv_performance(
                    f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
        elif environment == "Acrobot":
            data[algorithm] = get_csv_performance(
                f'../final_results/acrobot/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
            # data[algorithm] = get_csv_performance_2(
            #    f'../final_results/acrobot/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')

    # title = f'{environment} - Actor-Critics'
    title = f'{environment} - Baselines'
    # title = f'{environment} - Final Test Scores'
    # title = f'{environment} - n-step Q-Tsetlin-Machine'
    # title = f'{environment} - Q-Tsetlin-Machine'
    # title = 'Acrobot - Q-Tsetlin-Machine'
    # title = f'{environment} - Deep Q-Network'
    data, ratio = prune(data, new_size)
    plot_violin(data, title, ratio)


if __name__ == '__main__':
    ################## cartpole ##############################
    ################## cartpole ##############################
    ################## cartpole ##############################
    ################## cartpole ##############################
    """algorithms = {
        'DQN': {'folder': 'DQN', 'run': 'run_91', 'name': 'DQN'},
        'PPO': {'folder': 'PPO', 'run': 'run_3_final', 'name': 'PPO'},
        'TAC random': {'folder': 'TAC_random', 'run': 'run_2', 'name': 'TAC random'}
    }"""

    algorithms = {
        'TPPO': {'folder': 'TPPO', 'run': 'run_11', 'name': 'TPPO'},
        'TAC \n Update type a': {'folder': 'TAC_a', 'run': 'run_33', 'name': 'Tsetlin Actor-Critic - Type a update'},
        #'TAC \n Update type b': {'folder': 'TAC_b', 'run': 'run_9', 'name': 'Tsetlin Actor-Critic - Type b update'},
    }
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
        'TAC \n Update type a': {'folder': 'TAC_a', 'run': 'run_25', 'name': 'Tsetlin Actor-Critic - Type a update'},
        'TAC \n Update type b': {'folder': 'TAC_b', 'run': 'run_23', 'name': 'Tsetlin Actor-Critic - Type b update'},
    }"""

    """algorithms = {
        'QTM': {'folder': 'QTM', 'run': 'run_8', 'name': 'QTM'},
        'DQTM \n Update type a': {'folder': 'Double_QTM_a', 'run': 'run_14', 'name': 'DQTM - Type a update'},
        'DQTM \n Update type b': {'folder': 'Double_QTM_b', 'run': 'run_1b', 'name': 'DQTM - Type b update'},
    }"""

    """algorithms = {
        'n-step QTM': {'folder': 'n_step_QTM', 'run': 'run_1', 'name': 'n-step QTM'},
        'n-step DQTM \n Update type a': {'folder': 'n_step_Double_QTM_a', 'run': 'run_3', 'name': 'n-step DQTM - Type a update'},
        'n-step DQTM \n Update type b': {'folder': 'n_step_Double_QTM_b', 'run': 'run_4', 'name': 'n-step DQTM - Type b update'},
    }"""

    """algorithms = {
        'n-step DQN': {'folder': 'n_step_DQN', 'run': 'run_66', 'name': 'n-step DQN'},  # 20
        'PPO': {'folder': 'PPO', 'run': 'run_2', 'name': 'PPO'},
        'TAC random': {'folder': 'TAC_random', 'run': 'run_11', 'name': 'TAC random'},


        'QTM': {'folder': 'QTM', 'run': 'run_6', 'name': 'QTM'},
        'DQTM \n Update type a': {'folder': 'Double_QTM_a', 'run': 'run_12', 'name': 'DQTM - Type a update'},
        'DQTM \n Update type b': {'folder': 'Double_QTM_b', 'run': 'run_1b', 'name': 'DQTM - Type b update'},

        'n-step QTM': {'folder': 'n_step_QTM', 'run': 'run_1', 'name': 'n-step QTM'},
        'n-step DQTM \n Update type a': {'folder': 'n_step_Double_QTM_a', 'run': 'run_3',
                                         'name': 'n-step DQTM - Type a update'},
        'n-step DQTM \n Update type b': {'folder': 'n_step_Double_QTM_b', 'run': 'run_4',
                                         'name': 'n-step DQTM - Type b update'},
        'TPPO': {'folder': 'TPPO', 'run': 'run_6', 'name': 'TPPO'},
        'TAC \n Update type a': {'folder': 'TAC_a', 'run': 'run_5', 'name': 'Tsetlin Actor-Critic - Type a update'},
        'TAC \n Update type b': {'folder': 'TAC_b', 'run': 'run_13', 'name': 'Tsetlin Actor-Critic - Type b update'},
    }"""

    plot_many('Cartpole', algorithms, new_size=-1)
