import os
import matplotlib.pyplot as plt
from src.character_perturbation.text_perturbation import TextPerturbationHandler

def save_statistics(pathname, statistics):
    for stat in statistics:
        os.makedirs(pathname, exist_ok=True)
        with open(os.path.join(pathname, f'{stat}.txt'), 'w') as f:
            stats = '\n'.join(str(x) for x in statistics[stat])
            f.write(stats)


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
            s = f.read()
            plt.plot([float(x) for x in s.split('\n') if x != ''])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels)
    #plt.savefig(os.path.join(pathname, f'{title}.png'))
    plt.show()
    plt.close()

def save_perturbed_data(text: list, label: list, perturbation_weight: float = 1, pathname: str = 'data/perturbed_data'):
    perturbation_handler = TextPerturbationHandler()
    os.makedirs(pathname, exist_ok=True)
    perturbed_text = [perturbation_handler]
