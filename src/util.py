import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from src.character_perturbation.text_perturbation import TextPerturbationHandler
import seaborn as sns
import pandas as pd
import torch
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
    sns.set_theme()

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    for stat in statistics:
        with open(stat, 'r') as f:
            s = f.read()
            plt.plot([float(x) for x in s.split('\n') if x != ''])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(labels, loc='lower center')
    #plt.savefig(os.path.join(pathname, f'{title}.png'))
    plt.show()
    plt.close()


def load_state_fix_params(model, path):
    """
    This function is a hack to correctly load the state dict of models that were saved with an older version of the
    code so that we can evaluate them on test. I hate it as much as you do.

    Args: model: the model to load the state dict into
    path: the path to the state dict to load

    Returns:

    """
    try:
        model.load_state_dict(torch.load(
            path))
    except:
        # fix the state dict
        state_dict = torch.load('src/models/model.pt')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'encoder.elmo_embedding' in k:
                new_k = k.replace('encoder.elmo_embedding', 'encoder.elmo_embedder')
                new_state_dict[new_k] = v
            elif 'encoder.linear' in k:
                # the linear parameters don't do anything, ignore them; they're deleted in the latest model version.
                continue
            else:
                new_state_dict[k] = v
        if 'encoder.elmo_layernorm.weight' not in new_state_dict.keys():
            # add the useless layernorm parameters. Yes, it's a hack, but it makes the model load correctly.
            new_state_dict['encoder.elmo_layernorm.weight'] = torch.ones(768)
            new_state_dict['encoder.elmo_layernorm.bias'] = torch.zeros(768)
            model.encoder.layernorm_elmo_separately = False  # make sure the model doesn't try to actually use
            # these parameters since the model the state dict was saved from doesn't use them
        model.load_state_dict(new_state_dict)
