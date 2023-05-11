



if __name__ == '__main__':
    from src.util import *
    # plot training loss and accuracy for baseline model
    statistics = ['logs/experiments/ag_news_bert_baseline_2/training_loss.txt']
    labels = ['Training Loss']
    xlabel = 'Iteration'
    ylabel = 'Loss'

    plot_statistics_from_files(statistics,  labels,"Baseline Training Loss", xlabel, ylabel, './')

    statistics = ['logs/experiments/ag_news_bert_baseline_2/training_accuracy.txt']
    labels = ['Training Accuracy']
    plot_statistics_from_files(statistics,  labels,"Baseline Training Accuracy", xlabel, ylabel, './')

    statistics = ['logs/experiments/ag_news_bert_baseline_2/validation_accuracy.txt',
                  'logs/experiments/ag_news_bert_perturbed_2/validation_accuracy.txt',
                  'logs/experiments/ag_news_bert_concat_2/validation_accuracy.txt',
                  'logs/experiments/ag_news_bert_add_2/validation_accuracy.txt']
    labels = ['Baseline Validation Accuracy', 'Perturbed Validation Accuracy', 'Concat Validation Accuracy', 'Add Validation Accuracy']
    xlabel = 'Epoch'
    ylabel = 'Accuracy'
    plot_statistics_from_files(statistics,  labels, "Validation Accuracy", xlabel, ylabel, './')