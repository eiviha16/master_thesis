from matplotlib import pyplot as plt
import csv
import numpy as np


def plot(data, text, file_path):
    x = np.arange(1, len(data['mean']) + 1)

    plt.plot(x, data['mean'])
    plt.fill_between(x, np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']),
                     alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Rewards')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    plt.savefig(f'{file_path}/sample_plot.png')
    plt.show()


def plot_many(data, title, ratio):
    for key in data:
        x = np.arange(1, int((len(data[key]['mean'])) * ratio), step=ratio)
        plt.plot(x, data[key]['mean'], label=key)
        plt.fill_between(x, np.array(data[key]['mean']) - np.array(data[key]['std']),
                         np.array(data[key]['mean']) + np.array(data[key]['std']),
                         alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Rewards')
    plt.xlabel(f'Episodes')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'plots/rewards_comparison.png')
    plt.show()


def get_csv_performance(file_path):
    data = {'mean': [], 'std': []}
    with open(f'{file_path}/test_results.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'mean':
                data['mean'].append(float(row[0]))
                data['std'].append(float(row[1]))
        data['mean'] = data['mean'][:10000]
        data['std'] = data['std'][:10000]
    return data


def plot_test_results(file_path, text):
    data = get_csv_performance(file_path)
    plot(data, text, file_path)


def prune(data, new_size):
    for key in data:
        for m in data[key]:
            ratio = int(len(data[key][m]) / new_size)
            new_data = []
            for i in range(len(data[key][m])):
                if i % ratio == 0:
                    new_data.append(data[key][m][i])
            data[key][m] = new_data
    return data, ratio


def plot_many_rewards(algorithms, new_size):
    data = {}
    for algorithm in algorithms:
        data[algorithm] = get_csv_performance(f'../../cartpole_results/{algorithm}/{algorithms[algorithm]}')
    title = 'Cartpole'
    data, ratio = prune(data, new_size)
    plot_many(data, title, ratio)


if __name__ == "__main__":
    algorithms = {'n_step_Double_TMQN': 'run_35'}
    plot_many_rewards(algorithms, new_size=500)
