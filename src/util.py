import os
import matplotlib.pyplot as plt


def save_statistics(pathname, statistics):
    for stat in statistics:
        os.makedirs(pathname, exist_ok=True)
        with open(os.path.join(pathname, f'{stat}.txt'), 'w') as f:
            f.write(str(statistics[stat]))


def plot_statistics(statistics: tuple[list[float]], labels: list[str], title: str, xlabel: str, ylabel: str, pathname: str):
    os.makedirs(pathname, exist_ok=True)
    plt.figure(figsize=(10, 10))
    for stat in statistics:
        plt.plot(stat)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels)
    plt.savefig(os.path.join(pathname, f'{title}.png'))
    plt.close()

def plot_statistics_from_files(statistics: list[str], labels: list[str], title: str, xlabel: str, ylabel: str, pathname: str):
    os.makedirs(pathname, exist_ok=True)
    plt.figure(figsize=(10, 10))
    for stat in statistics:
        with open(stat, 'r') as f:
            plt.plot([float(x) for x in f.read().split('\n') if x])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels)
    plt.savefig(os.path.join(pathname, f'{title}.png'))
    plt.close()