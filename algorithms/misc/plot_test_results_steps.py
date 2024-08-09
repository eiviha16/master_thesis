from matplotlib import pyplot as plt
import csv
import numpy as np


def plot_many(names, data, title, ratio):
    plt.figure(figsize=(7, 5.5))
    for key in data:
        #x = np.arange(0, int((len(data[key]['mean'])) * ratio), step=ratio)
        x = np.arange(0, int(len(data[key]['mean']) * 50), 50)
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
        if environment == 'Cartpole':
            if algorithm == "DQN" and algorithm == "TAC random":
                data[algorithm] = get_csv_performance(
                    f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
            else:
                data[algorithm] = get_csv_performance(
                    f'../final_results/cartpole/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
        elif environment == "Acrobot":
            data[algorithm] = get_csv_performance(
                f'../final_results/acrobot/{algorithms[algorithm]["folder"]}/{algorithms[algorithm]["run"]}')
        names[algorithm] = algorithms[algorithm]["name"]
    title = f'{environment}'

    if new_size != -1:
        data, ratio = prune(data, new_size)
    else:
        ratio = 1
    plot_many(names, data, title, ratio)


if __name__ == "__main__":

    algorithms = {
        'DQN': {'folder': 'DQN', 'run': 'run_91', 'name': 'DQN'},
        'DQTM': {'folder': 'Double_QTM_a', 'run': 'run_4', 'name': 'Double QTM'},
        'TAC': {'folder': 'TAC_a', 'run': 'run_15', 'name': 'TAC'},
        'TPPO': {'folder': 'TPPO', 'run': 'run_11', 'name': 'TPPO'},
    }
    plot_many_rewards('Cartpole', algorithms, new_size=100)

