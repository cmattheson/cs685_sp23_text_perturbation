import torch

from src.training.training import *
from src.models.bert_models import *
from datasets import load_dataset
from src.data.datasets import *
from torch.utils.data import DataLoader
from src.util import *
from transformers import BertModel
from src.experiments.experiment import *
import os.path
def run_dummy_experiment() -> None:
    model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, 2))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1}
    tr = {'text': ['hello world', 'hello world'], 'label': [0, 1]}
    val = {'text': ['hello world', 'hello world'], 'label': [0, 1]}
    test = {'text': ['hello world', 'hello world'], 'label': [0, 1]}
    run_experiment('dummy', phases, model, optim, tr, val, test, save_datasets=True)

def run_baseline() -> None:
    """
    Run the baseline experiment on the ag_news dataset
    Returns: None

    """
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


def run_ag_news_experiments() -> None:
    """
    Run the AG News experiments. Uncomment the individual experimental setups to run them.
    Returns: None

    """
    # --------------------------------------------------------------------------------------------------------------------
    # do the concatenated model test
    # model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, num_classes))
    # optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}

    # TODO: also try separate layernorm here and see if validation accuracy can be improved

    # run_experiment('ag_news_concatenated_bert_elmo_model', phases, model, optim, train_data, test_data=test_data,
    #               train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, require_elmo_ids=True)

    # --------------------------------------------------------------------------------------------------------------------
    # do the additive model test
    # model = ClassifierModel(Bert_Plus_Elmo(), nn.Linear(768, num_classes))
    model = ClassifierModel(Bert_Plus_Elmo_Separate_Layernorm(), nn.Linear(768, num_classes))
    # optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    # optim = torch.optim.Adam(model.parameters(), lr=0.000003) # 10x smaller learning rate
    optim = torch.optim.Adam(model.parameters(), lr=0.000001)  # 30x smaller learning rate
    # phases = {'warmup': 1, 'elmo': 1, 'warmup': 1, 'elmo': 1, 'finetune': 5}  # add interleaved warmup and elmo phases
    # phases = {'warmup': 1, 'elmo': 1, 'finetune': 10}  # forget the interleaving: moaaaaar finetuning of full parameters
    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}  # use smaller number of epochs again, try separate layernorm

    # TODO: try doing separate layernorm for bert and elmo

    run_experiment('ag_news_additive_bert_elmo_model_separate_layernorm', phases, model, optim, train_data,
                   test_data=test_data,
                   train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, require_elmo_ids=True)


def run_ag_experiments_separate_layernorm() -> None:
    """

    Returns:
    TODO: implement this experiment
    """


# TODO: do some basic hyperparameter tuning. Try different learning rates, batch sizes, and number of epochs.

# TODO: run a baseline with word perturbation on validation and no perturbation on train

# TODO: run a baseline with word and char perturbation on validation and no perturbation on train

# TODO: implement and run experiments with word perturbation (no char perturbation)

# TODO: implement and run experiments using both word and char perturbation

if __name__ == '__main__':


    all_data = load_dataset('src/data/ag_news.py')
    num_classes = len(set(all_data['train']['label']))
    train_data = all_data['train']
    test_data = all_data['test']
    #run_dummy_experiment()


    run_ag_news_experiments()
