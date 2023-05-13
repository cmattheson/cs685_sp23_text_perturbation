from src.experiments.experiment import *
from src.models.bert_models import *
from src.training.training import *


def run_dummy_experiment() -> None:
    """
    Run a dummy experiment to test the experiment pipeline
    Returns:

    """
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

    run_experiment('ag_news_additive_bert_elmo_model_separate_layernorm', phases, model, optim, train_data,
                   test_data=test_data,
                   train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, require_elmo_ids=True)


def run_hyperparameter_optimization_concatenated_model() -> None:
    """
    TODO: decide whether to layernorm embeddings for bert and elmo separately or not
    Returns: None

    Run procedure to select hyperparameters for the concatenated model. In order to select the best perturbation
    rates, word and char perturbation rates will be optimized separately. The found optimal hyperparameters will be
    used to generate experimental results jointly. Although this approach is not necessarily optimal, it is hopefully
    a good approximation and will save a lot of time.

    """
    num_classes = 4
    model_constructor = Bert_Plus_Elmo_Concat
    val_perturbation_types = ['char', 'word']
    train_word_perturbation_rates = [0.0, 0.15, 0.3]
    val_word_perturbation_rate = 0.3
    char_perturbation_rates = [0.0, 1.0, 5.0]
    val_char_perturbation_rate = 5.0
    lrs = [0.001, 0.00001, 0.000001, 0.0000001]
    for lr in lrs:
        for val_perturbation_type in val_perturbation_types:
            if val_perturbation_type == 'char':
                for train_perturbation_rate in char_perturbation_rates:
                    name = f'ag_news_concatenated_bert_elmo_model_{val_perturbation_type}_{train_perturbation_rate}' \
                           f'_perturbed_hyperparameter_optimization_lr_{lr}'
                    model = ClassifierModel(model_constructor(), nn.Linear(768, num_classes))
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
                    run_experiment(name, phases, model, optim, train_data, test_data=test_data,
                                   train_char_perturbation_rate=train_perturbation_rate,
                                   val_char_perturbation_rate=val_char_perturbation_rate, require_elmo_ids=True)
            elif val_perturbation_type == 'word':
                for train_perturbation_rate in train_word_perturbation_rates:
                    name = f'ag_news_concatenated_bert_elmo_model_{val_perturbation_type}_{train_perturbation_rate}' \
                           f'_perturbed_hyperparameter_optimization_lr_{lr}'
                    model = ClassifierModel(model_constructor(), nn.Linear(768, num_classes))
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
                    run_experiment(name, phases, model, optim, train_data, test_data=test_data,
                                   train_word_perturbation_rate=train_perturbation_rate,
                                   val_word_perturbation_rate=val_word_perturbation_rate, require_elmo_ids=True)


def run_hyperparameter_optimization_additive_model() -> None:
    """
    TODO: decide whether to layernorm embeddings for bert and elmo separately or not
    Returns: None

    Run procedure to select hyperparameters for the additive model. In order to select the best perturbation
    rates, word and char perturbation rates will be optimized separately. The found optimal hyperparameters will be
    used to generate experimental results jointly. Although this approach is not necessarily optimal, it is hopefully
    a good approximation and will save a lot of time.
    """
    model_constructor = Bert_Plus_Elmo
    val_perturbation_types = ['char', 'word']
    train_word_perturbation_rates = [0.0, 0.15, 0.3]
    val_word_perturbation_rate = 0.3
    char_perturbation_rates = [0.0, 1.0, 5.0]
    val_char_perturbation_rate = 5.0
    lrs = [0.001, 0.00001, 0.000001, 0.0000001]
    for lr in lrs:
        for val_perturbation_type in val_perturbation_types:
            if val_perturbation_type == 'char':
                for train_perturbation_rate in char_perturbation_rates:
                    name = f'ag_news_concatenated_bert_elmo_model_{val_perturbation_type}_{train_perturbation_rate}' \
                           f'_perturbed_hyperparameter_optimization_lr_{lr}'
                    model = ClassifierModel(model_constructor(), nn.Linear(768, num_classes))
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
                    run_experiment(name, phases, model, optim, train_data, test_data=test_data,
                                   train_char_perturbation_rate=train_perturbation_rate,
                                   val_char_perturbation_rate=val_char_perturbation_rate, require_elmo_ids=True)
            elif val_perturbation_type == 'word':
                for train_perturbation_rate in train_word_perturbation_rates:
                    name = f'ag_news_concatenated_bert_elmo_model_{val_perturbation_type}_{train_perturbation_rate}' \
                           f'_perturbed_hyperparameter_optimization_lr_{lr}'
                    model = ClassifierModel(model_constructor(), nn.Linear(768, num_classes))
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
                    run_experiment(name, phases, model, optim, train_data, test_data=test_data,
                                   train_word_perturbation_rate=train_perturbation_rate,
                                   val_word_perturbation_rate=val_word_perturbation_rate, require_elmo_ids=True)



def run_ag_experiments_concat_model_separate_layernorm() -> None:
    """
    Test whether layer norm for bert and elmo separately improves performance for the concatenated model.
    Returns:
    """
    model = ClassifierModel(Bert_Plus_Elmo_Concat(layer_norm_elmo_separately=True), nn.Linear(768, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
    name = 'ag_news_concatenated_bert_elmo_model_separate_layernorm'
    run_experiment(name, phases, model, optim, train_data, test_data=test_data, require_elmo_ids=True)

    model = ClassifierModel(Bert_Plus_Elmo_Concat(layer_norm_elmo_separately=False), nn.Linear(768, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
    name = 'ag_news_concatenated_bert_elmo_model_layernorm_together'
    run_experiment(name, phases, model, optim, train_data, test_data=test_data, require_elmo_ids=True)


def run_ag_word_perturbation_experiments() -> None:
    """
    TODO: implement this
    Returns:

    """


# TODO: do some basic hyperparameter tuning. Try different learning rates, batch sizes, and number of epochs.

# TODO: run a baseline with word perturbation on validation and no perturbation on train

# TODO: run a baseline with word and word perturbation on validation and no perturbation on train

# TODO: implement and run experiments with word perturbation (no char perturbation)

# TODO: implement and run experiments using both word and char perturbation

# TODO: try doing separate layernorm for bert and elmo concatenated model


if __name__ == '__main__':
    all_data = load_dataset('src/data/ag_news.py')
    num_classes = len(set(all_data['train']['label']))
    train_data = all_data['train']
    test_data = all_data['test']
    # run_dummy_experiment()

    #run_ag_news_experiments()

    run_ag_experiments_concat_model_separate_layernorm()
