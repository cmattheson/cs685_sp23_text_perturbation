import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_hyperparam_optimization_accuracy():
    sns.set_theme()
    data = pd.DataFrame(columns=['experiment', 'epoch', 'accuracy'])

    for root, dirs, files in os.walk('logs/experiments/'):
        if 'hyperparameter' in root:
            for file in files:
                title = 'Validation Accuracy'
                if 'validation_accuracy' in file:
                    with open(root + '/' + file, 'r') as f:
                        s = f.read()
                        print(root)
                        experiment = root[root.rfind('/ag_news') + 1:]
                        data = pd.concat([data, pd.DataFrame({'experiment': [experiment] * len(s.split('\n')), 'epoch':
                            list(range(len(s.split('\n')))), 'accuracy': [float(x) for x in s.split('\n') if x != '']})], ignore_index=True)
    print(data)
    sns.relplot(data=data, x='epoch', y='accuracy', hue='experiment', kind='line', height=8, aspect=1.5)
    plt.show()
    #plot_statistics_from_files(statistics, labels, title, xlabel, ylabel, 'logs/figures/')



if __name__ == '__main__':
    from src.util import *
    # plot training loss and accuracy for baseline model
    plot_hyperparam_optimization_accuracy()
