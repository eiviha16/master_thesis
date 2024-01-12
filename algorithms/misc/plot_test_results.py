from matplotlib import pyplot as plt
import csv
import numpy as np


def plot(data, text, file_path):
    x = np.arange(1, len(data['timesteps']) + 1)

    plt.plot(x, data['mean'])
    # plt.plot(data['timesteps'], data['mean'])
    plt.fill_between(x, np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']),
                     alpha=0.25)
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.ylabel(f'Rewards')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    plt.savefig(f'{file_path}/sample_plot.png')
    # plt.show()


def plot_feedback(data, text, file_path):
    for key in data:
        if key != 'timesteps':
            norm = (data[key] - np.min(data[key])) / (np.max(data[key]) - np.min(data[key]))
            x = np.arange(0, len(norm))
            x *= 100
            plt.plot(x, norm, label=key)
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["Type I 0", '0.25', '0.5', '0.75', 'Type II 1'])
    plt.ylabel(f'Ratio')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f'{file_path}/feedback_plot.png')
    # plt.show()


def plot_actions(data, text, file_path):
    # for key in data:
    #    if key != 'timesteps':
    # norm = (data[key] - np.min(data[key])) / (np.max(data[key]) - np.min(data[key]))
    norm = (data['ratio'] - np.min(data['ratio'])) / (np.max(data['ratio']) - np.min(data['ratio']))

    x = np.arange(0, len(data['ratio']))
    x = x * 100
    plt.plot(x, norm)
    # plt.fill_between(data['timesteps'], np.array(data['mean']) - np.array(data['std']), np.array(data['mean']) + np.array(data['std']), alpha=0.25)
    plt.gca().yaxis.grid(True, linestyle='dashed')
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ["TM1 0", '0.25', '0.5', '0.75', 'TM2 1'])
    plt.ylabel(f'Ratio')
    plt.xlabel(f'Episodes')
    plt.title(f'{text["title"]}')
    # plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f'{file_path}/actions_plot.png')


def get_ratio(data):
    _feedback = {'TM 1': [], 'TM 2': [], 'timesteps': []}
    for i in range(len(data['1_typeI'])):
        if i % 100 == 0:
            if data['1_typeI'][i] != 0:
                _feedback['TM 1'].append(data['1_typeII'][i] / data['1_typeI'][i])
            else:
                _feedback['TM 1'].append(data['1_typeII'][i] / 1)

            if data['2_typeI'][i] != 0:
                _feedback['TM 2'].append(data['2_typeII'][i] / data['2_typeI'][i])
            else:
                _feedback['TM 2'].append(data['2_typeII'][i] / 1)
            _feedback['timesteps'].append(data['timesteps'][i])
    for key in _feedback:
        _feedback[key] = np.array(_feedback[key])
    return _feedback


def get_action_ratio(data):
    new_data = {'ratio': [], 'timesteps': []}
    for i in range(len(data['tm1'])):
        if i % 100 == 0:
            if data['tm1'][i] != 0:
                new_data['ratio'].append(data['tm2'][i] / data['tm1'][i])
            else:
                new_data['ratio'].append(data['tm2'][i] / 1)
            new_data['timesteps'].append(data['timesteps'][i])
    return new_data


def get_csv_performance(file_path):
    data = {'mean': [], 'std': [], 'timesteps': []}
    with open(f'{file_path}/test_results.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'mean':
                data['mean'].append(float(row[0]))
                data['std'].append(float(row[1]))
                data['timesteps'].append(float(row[2]))
    return data


def get_csv_feedback(file_path):
    # 1_typeI, 1_typeII, 2_typeI, 2_typeII, steps

    data = {'1_typeI': [], '1_typeII': [], '2_typeI': [], '2_typeII': [], 'timesteps': []}
    with open(f'{file_path}/feedback.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != '1_typeI':
                data['1_typeI'].append(float(row[0]))
                data['1_typeII'].append(float(row[1]))
                data['2_typeI'].append(float(row[2]))
                data['2_typeII'].append(float(row[3]))
                data['timesteps'].append(float(row[4]))
    return data


def get_actions(file_path):
    # 1_typeI, 1_typeII, 2_typeI, 2_typeII, steps

    data = {'tm1': [], 'tm2': [], 'timesteps': []}
    with open(f'{file_path}/actions.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'tm1':
                data['tm1'].append(float(row[0]))
                data['tm2'].append(float(row[1]))
                data['timesteps'].append(float(row[2]))
    return data


def plot_test_results(file_path, text):
    data = get_csv_performance(file_path)
    plot(data, text, file_path)


def feedback(file_path, text):
    data = get_csv_feedback(file_path)
    data = get_ratio(data)
    plot_feedback(data, text, file_path)


def actions(file_path, text):
    data = get_actions(file_path)
    data = get_action_ratio(data)
    plot_actions(data, text, file_path)


if __name__ == "__main__":
    text = {'title': 'Feedback TMQN'}
    feedback('../results/TMQN_w_feedback_balance/run_26', text)
