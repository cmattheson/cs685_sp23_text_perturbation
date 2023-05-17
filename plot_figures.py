import os
from typing import Iterable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_hyperparam_optimization_accuracy(keywords, title, save_path=None, hue='lr'):
    sns.set_theme()
    data = pd.DataFrame()
    experiment_name = ', '.join(hue) if isinstance(hue, Iterable) and not isinstance(hue, str) else hue
    if isinstance(keywords, str):
        keywords = [keywords]
    for root, dirs, files in os.walk('logs/experiments/'):
        if all([k in root for k in keywords]):
            for file in files:
                if 'validation_accuracy' in file:
                    with open(root + '/' + file, 'r') as f:
                        s = f.read()
                        experiment = []
                        for x in hue:
                            start = root.rfind(x) + len(x) + 1
                            end = root.find('_', start) if '_' in root[start:] else len(root)
                            experiment.append(root[start:end])
                        name = ' '.join(experiment)
                        #print(experiment)
                        data = pd.concat([data, pd.DataFrame({experiment_name: [', '.join(experiment)] * len(s.split('\n')), 'epoch':
                            list(range(len(s.split('\n')))), 'accuracy': [float(x) for x in s.split('\n') if
                                                                          x != '']})], ignore_index=True)
    print(data)
    g = sns.relplot(data=data, x='epoch', y='accuracy', hue=experiment_name, kind='line', height=8, aspect=1)
    g.fig.subplots_adjust(top=0.9)
    plt.title(title)
    if save_path:
        g.savefig(save_path)
    plt.show()
    #plot_statistics_from_files(statistics, labels, title, xlabel, ylabel, 'logs/figures/')

def plot_final_validation_accuracy(experiment_type, save_path=None, hue='lr'):
    sns.set_theme()
    data = pd.DataFrame()
    for root, dirs, files in os.walk('logs/experiments'):
        if experiment_type in root:
            for file in files:
                if 'validation_accuracy' in file:
                    with open(root + '/' + file, 'r') as f:
                        s = f.read()
                        start = root.rfind(hue) + len(hue)+1
                        end = root.rfind('_', start) if '_' in root[start:] else len(root)
                        experiment = root[start:end]
                        data = pd.concat([data, pd.DataFrame({hue: experiment, 'accuracy': [[float(x) for x in s.split('\n') if x != ''][-1]]})], ignore_index=True)
    print(data)
    g = sns.catplot(data=data, x=hue, y='accuracy', kind='bar', height=8, aspect=1)
    g.fig.subplots_adjust(top=0.9)
    plt.title('Validation Accuracy of concatenated model for with different char perturbation rates and learning rates')
    if save_path:
        g.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    from src.util import *

    # baseline model hyperparameters
    plot_hyperparam_optimization_accuracy(['baseline_model_lr'],'Validation Accuracy vs Epoch for baseline model with different learning rates', save_path='logs/figures/hyperparameter_baseline.png', hue=['lr'])

    # concatenated model hyperparameters
    plot_hyperparam_optimization_accuracy(['hyperparameter', 'concatenated', 'char'],'Validation Accuracy vs Epoch for concated model with different learning rates and char perturbation', save_path='logs/figures/hyperparameter_char_concat.png', hue=['char', 'lr'])
    #plot_hyperparam_optimization_accuracy(['hyperparameter', 'concatenated', 'word'],'Validation Accuracy vs Epoch for concated model with different learning rates and word perturbation', save_path='logs/figures/hyperparameter_word_concat.png', hue=['word', 'lr'])

    # additive model hyperparameters
    plot_hyperparam_optimization_accuracy(['hyperparameter', 'additive', 'char'],'Validation Accuracy vs Epoch for additive model with different learning rates and char perturbation', save_path='logs/figures/hyperparameter_char_additive.png', hue=['char', 'lr'])
    plot_hyperparam_optimization_accuracy(['hyperparameter', 'additive', 'word'],'Validation Accuracy vs Epoch for additive model with different learning rates and word perturbation', save_path='logs/figures/hyperparameter_word_additive.png', hue=['word', 'lr'])

    #plot_hyperparam_optimization_accuracy('baseline_model_lr', save_path='logs/figures/baseline_model_lr.png')
    #plot_final_validation_accuracy('baseline_model_lr', save_path='logs/figures/baseline_model_lr_final.png')
    #plot_hyperparam_optimization_accuracy(['final', 'baseline', 'char'], save_path='logs/figures/hyperparameter.png', hue=['char', 'lr'])
