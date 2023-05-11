import os
import matplotlib.pyplot as plt
from src.character_perturbation.text_perturbation import TextPerturbationHandler

def save_statistics(pathname:str, statistics:dict[str, list[float]]):
    """

    Args:
        pathname: where to save
        statistics: dict of statistics to save where key is the name of the statistic and value is a list of values,
        usually one for each epoch

    Returns:

    """
    for stat in statistics:
        os.makedirs(pathname, exist_ok=True)
        with open(os.path.join(pathname, f'{stat}.txt'), 'w') as f:
            if isinstance(statistics[stat], list):
                # if the statistic is a list, write each value on a new line
                f.write('\n'.join(str(x) for x in statistics[stat]))
            else:
                # otherwise just write the value
                f.write(str(statistics[stat]))



def plot_statistics(statistics: tuple[list[float]], labels: list[str], title: str, xlabel: str, ylabel: str, pathname: str):
    """

    Args:
        statistics:
        labels:
        title:
        xlabel:
        ylabel:
        pathname:

    Returns:

    """
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
