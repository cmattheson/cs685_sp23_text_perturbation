import os

def plot_hyperparam_optimization_accuracy():
    statistics = []
    labels = []
    xlabel = 'Epoch'
    ylabel = 'Accuracy'

    for root, dirs, files in os.walk('logs/experiments/'):
        if 'hyperparameter' in root:
            for file in files:
                title = 'Validation Accuracy'
                if 'validation_accuracy' in file:
                    labels.append(root[root.rfind('/ag_news') + 1:])
                    statistics.append(root + '/' + file)
    plot_statistics_from_files(statistics, labels, title, xlabel, ylabel, 'logs/figures/')



if __name__ == '__main__':
    from src.util import *
    # plot training loss and accuracy for baseline model
    plot_hyperparam_optimization_accuracy()
