
# external libraries
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple

# other project modules
from src.character_perturbation.text_perturbation import TextPerturbationHandler


def format_statistic_to_string(statistic: Union[float, List[float]]) -> str:

    if isinstance(statistic, list):
        formatted_stat = '\n'.join(str(x) for x in statistic)
    else:
        formatted_stat = str(statistic)

    return formatted_stat


def save_statistics(pathname: str, statistics: Dict[str, List[float]]):
    """

    Args:
        pathname: where to save
        statistics: dict of statistics to save where key is the name of the statistic
        and value is a list of values, usually one for each epoch

    Returns:

    """
    for stat in statistics.keys():
        os.makedirs(pathname, exist_ok=True)
        formatted_stat = format_statistic_to_string(statistics.get(stat))
        with open(f'{pathname}/{stat}.txt', 'w') as file:
            file.write(formatted_stat)


def plot_statistics(
        statistics: Tuple[List[float]],
        labels: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        pathname: str
):

    os.makedirs(pathname, exist_ok=True)
    plt.figure(figsize=(10, 10))
    for stat in statistics:
        plt.plot(stat)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels)
    plt.savefig(f'{pathname}/{title}.png')
    plt.close()


def plot_statistics_from_files(
    statistics: List[str],
    labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    pathname: str
):
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
    plt.show()
    plt.close()
