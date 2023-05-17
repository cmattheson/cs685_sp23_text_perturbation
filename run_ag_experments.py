from os import makedirs

from src.experiments.experiment import *
from src.models.bert_models import *
from src.training.training import *
from transformers import BertModel
from src.util import *


def run_dummy_experiment() -> None:
    """
    Run a dummy experiment to test the experiment pipeline
    Returns:

    """
    model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, 4))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1}
    tr = {'text': ['hello world', 'hello world'], 'label': [0, 1]}
    val = {'text': ['hello world', 'hello world'], 'label': [0, 1]}
    test = {'text': ['hello world', 'hello world'], 'label': [0, 1]}
    run_experiment('dummy', phases, model, optim, tr, val, test, save_datasets=True)


def train_final_concat_model_char(train_data):
    char_perturbation_rate = 5.0
    lr = 3e-5
    name = f'char-perturbation_{char_perturbation_rate}_final_experiment'
    model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, 4))
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    run_experiment(name, {'warmup': 1, 'elmo': 1, 'finetune': 5}, model, optim, train_data=train_data, split=1.0,
                   train_char_perturbation_rate=char_perturbation_rate, require_elmo_ids=True)


def train_final_concat_model_word(train_data):
    word_perturbation_rate = 0.3
    lr = 3e-5
    name = f'word-perturbation_{word_perturbation_rate}_final_experiment'
    model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, 4))
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    run_experiment(name, {'warmup': 1, 'elmo': 1, 'finetune': 5}, model, optim, train_data=train_data, split=1.0,
                   train_word_perturbation_rate=word_perturbation_rate, require_elmo_ids=True)


def train_final_concat_model_both(train_data):
    char_perturbation_rate = 5.0
    word_perturbation_rate = 0.3
    lr = 3e-5
    name = f'word-perturbation_{word_perturbation_rate}_final_experiment'
    model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, 4))
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    run_experiment(name, {'warmup': 1, 'elmo': 1, 'finetune': 5}, model, optim, train_data=train_data, split=1.0,
                   train_char_perturbation_rate=char_perturbation_rate,
                   train_word_perturbation_rate=word_perturbation_rate, require_elmo_ids=True)

def train_final_baseline_models_char():
    # done 1
    # char_perturbation_rates = [1, 3, 5]
    char_perturbation_rates = [3, 5]
    for char_perturbation_rate in char_perturbation_rates:
        name = f'char-perturbation_{char_perturbation_rate}_final_baseline'
        model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, 4))
        optim = torch.optim.Adam(model.parameters(), lr=0.00003)
        run_experiment(name, {'warmup': 1, 'finetune': 5}, model, optim, train_data, split=1.0,
                       train_char_perturbation_rate=char_perturbation_rate)


def train_final_baseline_models_word():
    word_perturbation_rates = [0.15, 0.3, 0.5]
    for word_perturbation_rate in word_perturbation_rates:
        name = f'word-perturbation_{word_perturbation_rate}_final_baseline'
        model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, 4))
        optim = torch.optim.Adam(model.parameters(), lr=0.00003)
        run_experiment(name, {'warmup': 1, 'finetune': 5}, model, optim, train_data, split=1.0,
                       train_word_perturbation_rate=word_perturbation_rate)


def train_final_baseline_models_char_word():
    char_perturbation_rate = 5
    word_perturbation_rate = 0.3
    name = f'word-perturbation_{word_perturbation_rate}_char-perturbation_{char_perturbation_rate}_final_baseline'
    model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, 4))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    run_experiment(name, {'warmup': 1, 'finetune': 5}, model, optim, train_data, split=1.0,
                   train_char_perturbation_rate=char_perturbation_rate,
                   train_word_perturbation_rate=word_perturbation_rate)


def run_baseline() -> None:
    """
    Run the baseline experiment on the ag_news dataset. This runs the learning rate hyperparameter optimization tests
    for baseline models.

    Returns: None

    """
    # lrs = [0.00003, 0.00001, 0.000003, 0.000001]
    lrs = [0.000003, 0.000001]  # run with the last two learning rates (crashed partway through because I was stupid
    # and ran it while testing another experiment)

    for lr in lrs:
        name = f'ag_news_baseline_model_lr_{lr}'
        model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, num_classes))
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        phases = {'warmup': 1, 'finetune': 5}

        run_experiment(name, phases, model, optim, train_data)


def eval_baseline() -> None:
    pass


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


def run_final_ag_news_experiments() -> None:
    """
    Run the AG News experiments. Uncomment the individual experimental setups to run them.
    Returns: None

    """
    # --------------------------------------------------------------------------------------------------------------------
    # do the concatenated model test
    # model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, num_classes))
    # optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    phases = {'warmup': 1, 'elmo': 1, 'finetune': 10}

    # TODO: set up final hyperparameters for the concatenated model with char perturbation
    name = 'ag_news_concatenated_bert_elmo_model_char_perturbation_final'
    lr = 0.000001
    train_char_perturbation_rate = 5.0
    val_char_perturbation_rate = 5.0
    model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    run_experiment(name, phases, model, optim, train_data,
                   test_data=test_data,
                   train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, require_elmo_ids=True)

    # TODO: set up final hyperparameters for the concatenated model with word perturbation
    name = 'ag_news_concatenated_bert_elmo_model_word_perturbation_final'
    run_experiment(name, phases, model, optim, train_data,
                   test_data=test_data,
                   train_char_perturbation_rate=5.0, val_char_perturbation_rate=5.0, require_elmo_ids=True)

    # TODO: set up final hyperparameters for the concatenated model with char and word perturbation
    name = 'ag_news_concatenated_bert_elmo_model_char_and_word_perturbation_final'

    run_experiment(name, phases, model, optim, train_data,
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
    print('Running hyperparameter optimization for concatenated model')
    num_classes = 4
    model_constructor = Bert_Plus_Elmo_Concat
    # val_perturbation_types = ['char', 'word']
    val_perturbation_types = ['char']

    train_word_perturbation_rates = [0.15, 0.3]
    val_word_perturbation_rate = 0.3
    char_perturbation_rates = [1.0, 5.0]
    val_char_perturbation_rate = 5.0
    lrs = [0.00003]
    for lr in lrs:
        for val_perturbation_type in val_perturbation_types:
            if val_perturbation_type == 'char':
                # test with char perturbations
                for train_perturbation_rate in char_perturbation_rates:
                    print(
                        f'Running hyperparameter optimization for char perturbation concatenated model with lr {lr}, perturbation rate {train_perturbation_rate}')
                    name = f'ag_news_concatenated_bert_elmo_model_{val_perturbation_type}_{train_perturbation_rate}' \
                           f'_perturbed_hyperparameter_optimization_lr_{lr}'
                    model = ClassifierModel(model_constructor(), nn.Linear(768, num_classes))
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
                    run_experiment(name, phases, model, optim, train_data,
                                   train_char_perturbation_rate=train_perturbation_rate,
                                   val_char_perturbation_rate=val_char_perturbation_rate,
                                   require_elmo_ids=True)
            elif val_perturbation_type == 'word':
                # test with word perturbations
                for train_perturbation_rate in train_word_perturbation_rates:
                    print(
                        f'Running hyperparameter optimization for word perturbation concatenated model with lr {lr}, perturbation rate {train_perturbation_rate}')

                    name = f'ag_news_concatenated_bert_elmo_model_{val_perturbation_type}_{train_perturbation_rate}' \
                           f'_perturbed_hyperparameter_optimization_lr_{lr}'
                    model = ClassifierModel(model_constructor(), nn.Linear(768, num_classes))
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
                    run_experiment(name, phases, model, optim, train_data,
                                   train_word_perturbation_rate=train_perturbation_rate,
                                   val_word_perturbation_rate=val_word_perturbation_rate, require_elmo_ids=True)


def run_hyperparameter_optimization_additive_model() -> None:
    """
    Returns: None

    Run procedure to select hyperparameters for the additive model. In order to select the best perturbation
    rates, word and char perturbation rates will be optimized separately. The found optimal hyperparameters will be
    used to generate experimental results jointly. Although this approach is not necessarily optimal, it is hopefully
    a good approximation and will save a lot of time.
    """
    print('Running hyperparameter optimization for additive model')
    num_classes = 4
    model_constructor = Bert_Plus_Elmo_Separate_Layernorm
    # val_perturbation_types = ['char', 'word']
    val_perturbation_types = ['char']

    # val_perturbation_types = ['char']
    # val_perturbation_types = ['word']
    train_word_perturbation_rates = [0.15, 0.5]
    val_word_perturbation_rate = 0.3
    char_perturbation_rates = [1.0, 5.0]
    val_char_perturbation_rate = 5.0
    lrs = [0.000003]
    for lr in lrs:
        for val_perturbation_type in val_perturbation_types:
            if val_perturbation_type == 'char':
                # test with char perturbations
                for train_perturbation_rate in char_perturbation_rates:
                    print(
                        f'Running hyperparameter optimization for char perturbation additive model with lr {lr}, perturbation rate {train_perturbation_rate}')

                    name = f'ag_news_additive_bert_elmo_model_separate_layernorm_{val_perturbation_type}_{train_perturbation_rate}' \
                           f'_perturbed_hyperparameter_optimization_lr_{lr}'
                    model = ClassifierModel(model_constructor(), nn.Linear(768, num_classes))
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    phases = {'warmup': 1, 'elmo': 1, 'finetune': 5}
                    run_experiment(name, phases, model, optim, train_data, test_data=test_data,
                                   train_char_perturbation_rate=train_perturbation_rate,
                                   val_char_perturbation_rate=val_char_perturbation_rate, require_elmo_ids=True)
            elif val_perturbation_type == 'word':
                # test with word perturbations
                for train_perturbation_rate in train_word_perturbation_rates:
                    print(
                        f'Running hyperparameter optimization for word perturbation additive model with lr {lr}, perturbation rate {train_perturbation_rate}')
                    name = f'ag_news_additive_bert_elmo_model_separate_layernorm_{val_perturbation_type}_{train_perturbation_rate}' \
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


def evaluate_models():
    models = [
        # commented out are already run
        # concatenated models
        # something is screwey with these models, use the old ones instead
        # 'ag_news_concatenated_bert_elmo_model_char_1.0_perturbed_hyperparameter_optimization_lr_3e-05',
        # 'ag_news_concatenated_bert_elmo_model_char_5.0_perturbed_hyperparameter_optimization_lr_3e-05',
        # try this with one of the other saved models

        # additive models
        # 'ag_news_additive_bert_elmo_model_char_1.0_perturbed_hyperparameter_optimization_lr_3e-05',
        'char-perturbation_5_final_baseline',
        'ag_news_concatenated_bert_elmo_model_separate_layernormchar_5.0_perturbed_hyperparameter_optimization_lr_3e-06',
        'ag_news_additive_bert_elmo_model_char_5.0_perturbed_hyperparameter_optimization_lr_3e-05',
        # basline models
        'char-perturbation_1_final_baseline',
        'char-perturbation_3_final_baseline',
        'char-perturbation_5_final_baseline',
        'word-perturbation_0.15_final_baseline',
        'word-perturbation_0.3_final_baseline',
        'word-perturbation_0.5_final_baseline',
        'word-perturbation_0.3_char_perturbation_5_final_baseline',
    ]
    # test_char_perturbation_rates = [0, 1.0, 5.0]
    test_char_perturbation_rates = [5.0]
    test_word_perturbation_rates = [0, 0.3, 0.5]
    test_data = pd.read_csv('src/data/datasets/ag_news_cleaned_test.csv')
    for model_str in models:
        print(model_str)
        # model = ClassifierModel(Bert_Plus_Elmo_Separate_Layernorm(), nn.Linear(768, 4))
        model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, 4))

        load_state_fix_params(model, f'src/models/pretrained/{model_str}/model.pt')
        model.to('cuda')
        for test_word_perturbation_rate in test_word_perturbation_rates:
            for test_char_perturbation_rate in test_char_perturbation_rates:
                test_set = PerturbedSequenceDataset(test_data['text'],
                                                    torch.tensor(test_data['label']).to(torch.long),
                                                    val_word_perturbation_rate=test_word_perturbation_rate,
                                                    val_char_perturbation_rate=test_char_perturbation_rate,
                                                    require_elmo_ids=False)
                test_set.eval()
                test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
                criterion = torch.nn.functional.cross_entropy
                loss, accuracy = compute_statistics(model, criterion, test_loader)
                makedirs(f'logs/experiments/test/{model_str}/', exist_ok=True)
                with open(
                        f'logs/experiments/test/{model_str}/char_{test_char_perturbation_rate}_word_{test_word_perturbation_rate}_test_results_.txt',
                        'w') as f:
                    f.write(f'{loss}, {accuracy}')


if __name__ == '__main__':
    all_data = load_dataset('src/data/ag_news.py')
    num_classes = len(set(all_data['train']['label']))
    train_data = all_data['train']
    #test_data = all_data['test']
    # run_dummy_experiment()

    # run_ag_news_experiments()

    # run_ag_experiments_concat_model_separate_layernorm()

    # run_hyperparameter_optimization_concatenated_model()

    # run_hyperparameter_optimization_additive_model()

    # run_baseline()

    # train_final_baseline_models_char()
    # train_final_baseline_models_word()
    # train_final_baseline_models_char_word()
    train_final_concat_model_char(train_data)
    train_final_concat_model_word(train_data)
    train_final_concat_model_both(train_data)
    #evaluate_models()
