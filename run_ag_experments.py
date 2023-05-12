from src.training.training import *
from src.models.bert_models import *
from datasets import load_dataset
from src.data.datasets import *
from torch.utils.data import DataLoader
from src.util import *
from transformers import BertModel
from src.experiments.experiment import *


def run_baseline():
    model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1, 'finetune': 5}

    # train set with no perturbation on train and 5 char perterbation on val and test
    run_experiment('ag_news_baseline_model', phases, model, optim, train_data, test_data=test_data,
                   val_char_perturbation_rate=5.0)
    # --------------------------------------------------------------------------------------------------------------------
    # do the perturbed test
    model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)

    run_experiment('ag_news_baseline_model_with_train_perturbation', phases, model, optim, train_data,
                   test_data=test_data, train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, )


def run_ag_news_experiments():
    # --------------------------------------------------------------------------------------------------------------------
    # do the concatenated model test
    #model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, num_classes))
    #optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}

    #run_experiment('ag_news_concatenated_bert_elmo_model', phases, model, optim, train_data, test_data=test_data,
    #               train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, require_elmo_ids=True)

    # --------------------------------------------------------------------------------------------------------------------
    # do the additive model test
    model = ClassifierModel(Bert_Plus_Elmo(), nn.Linear(768, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    optim = torch.optim.Adam(model.parameters(), lr=0.000003) # 10x smaller learning rate


    run_experiment('ag_news_additive_bert_elmo_model_lr_000003', phases, model, optim, train_data, test_data=test_data,
                   train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, require_elmo_ids=True)


if __name__ == '__main__':
    all_data = load_dataset('src/data/ag_news.py')
    num_classes = len(set(all_data['train']['label']))
    train_data = all_data['train']
    test_data = all_data['test']
    """
    This script runs all the experiments for the ag_news dataset
    uncomment individual experiments to run them
    """

    #run_baseline()
    # run_baseline_perturbed_test()
    run_ag_news_experiments()
