from src.training.training import *
from src.models.bert_models import *
from datasets import load_dataset
from src.data.datasets import *
from torch.utils.data import DataLoader
from src.util import *
from transformers import BertModel
import datetime as dt

# TODO: where are we getting all_data from?


def train_and_test_model(
        log_directory: str,
        model_directory: str,
        model_name: str,
        encoder,
        perturbation_rate: Union[int, float],
        require_elmo_ids: bool
):

    num_classes = len(set(all_data['train']['label']))
    model = ClassifierModel(
        encoder=encoder, classifier=nn.Linear(768, num_classes)
    )

    training_data = PerturbedSequenceDataset(
        all_data['train']['text'],
        torch.tensor(all_data['train']['label']),
        require_elmo_ids=require_elmo_ids,
        train_char_perturbation_rate=perturbation_rate,
        word_perturbation_rate=perturbation_rate
    )
    train_set, val_set = train_val_test_split(training_data, pct_train=0.8, pct_val=0.2)
    test_set = PerturbedSequenceDataset(
        all_data['test']['text'],
        torch.tensor(all_data['test']['label']),
        require_elmo_ids=False
    )
    test_set.eval()

    criterion = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    train_loader = DataLoader(
        train_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True
    )
    statistics = train(
        model,
        train_loader,
        criterion,
        optim,
        device='cuda',
        val_loader=val_loader,
        test_loader=test_loader,
        phases=phases,
        record_training_statistics=True,
        model_save_path=f'{model_directory}/{model_name}.pt',
        record_time_statistics=True)
    save_statistics(f'{log_directory}/{model_name}', statistics)


def run_baseline(phases, dataset_name: str, reference_ts: dt.datetime = None):

    if reference_ts is None:
        reference_ts = dt.datetime.now()
    unix_ts = int(reference_ts.timestamp())
    runtime_folder_name = f'text_perturbation_test_{dataset_name}_{unix_ts}'

    test_name = f'{dataset_name}_bert_baseline'
    # with 1 epoch warmup and elmo, 5 epoch fine tune, 5x char perturbation weight
    model = ClassifierModel(
        BertModel.from_pretrained('bert-base-uncased'),
        nn.Linear(768, num_classes)
    )

    # train set with no perturbation
    full_dataset = PerturbedSequenceDataset(
        all_data['train']['text'],
        torch.tensor(all_data['train']['label']),
        train_char_perturbation_rate=0,
        require_elmo_ids=False,
        word_perturbation_rate=0.0
    )
    train_set, val_set = train_val_test_split(full_dataset, pct_train=0.8, pct_val=0.2)
    test_set = PerturbedSequenceDataset(
        all_data['test']['text'],
        torch.tensor(all_data['test']['label']),
        require_elmo_ids=False
    )
    test_set.eval()

    criterion = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    train_loader = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    statistics = train(
        model,
        train_loader,
        criterion,
        optim,
        device='cuda',
        val_loader=val_loader,
        test_loader=test_loader,  # aaargh remember to uncomment this next time
        phases=phases,
        record_training_statistics=True,
        model_save_path=f'src/models/pretrained/{test_name}_model.pt',
        record_time_statistics=True)
    save_statistics(f'logs/experiments/{runtime_folder_name}/{test_name}', statistics)

    test_name = f'{dataset_name}_bert_perturbed'
    model = ClassifierModel(
        BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, num_classes)
    )

    full_dataset = PerturbedSequenceDataset(
        all_data['train']['text'],
        torch.tensor(all_data['train']['label']),
        require_elmo_ids=False,
        train_char_perturbation_rate=5.0
    )
    train_set, val_set = train_val_test_split(full_dataset, pct_train=0.8, pct_val=0.2)
    test_set = PerturbedSequenceDataset(all_data['test']['text'], torch.tensor(all_data['test']['label']),
                                        require_elmo_ids=False)
    test_set.eval()

    criterion = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    train_loader = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    statistics = train(
        model,
        train_loader,
        criterion,
        optim,
        device='cuda',
        val_loader=val_loader,
        test_loader=test_loader,
        phases=phases,
        record_training_statistics=True,
        model_save_path=f'src/models/pretrained/{test_name}.pt',
        record_time_statistics=True)
    save_statistics(f'logs/experiments/{runtime_folder_name}/{test_name}', statistics)


def run_ag_news_experiments(phases):
    # uses bert_plus_elmo, bert_plus_elmo_concat
    test_name = 'ag_news_bert_concat'
    model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, num_classes))

    train_set = PerturbedSequenceDataset(all_data['train']['text'],
                                         torch.tensor(all_data['train']['label']), require_elmo_ids=True, train_char_perturbation_rate=5.0)
    train_set, val_set = train_val_test_split(train_set, pct_train=0.8, pct_val=0.2)
    test_set = PerturbedSequenceDataset(
        all_data['test']['text'], torch.tensor(all_data['test']['label']),
                                        require_elmo_ids=True)
    test_set.eval()

    train_loader = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    criterion = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)

    statistics = train(model, train_loader, criterion, optim, device='cuda',
                       val_loader=val_loader,
                       test_loader=test_loader,
                       phases=phases,
                       record_training_statistics=True,
                       model_save_path=f'src/models/pretrained/{test_name}_model.pt',
                       record_time_statistics=True)
    save_statistics(f'logs/experiments/{test_name}', statistics)

    # --------------------------------------------------------------------------------------------------------------------
    test_name = 'ag_news_bert_add'
    test_name = 'ag_news_bert_add_2' # with 1 epoch warmup and elmo, 5 epoch fine tune
    model = ClassifierModel(Bert_Plus_Elmo(), nn.Linear(768, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)

    statistics = train(model, train_loader, criterion, optim, device='cuda',
                       val_loader=val_loader,
                       test_loader=test_loader,
                       phases=phases,
                       record_training_statistics=True,
                       model_save_path=f'src/models/pretrained/{test_name}_model.pt',
                       record_time_statistics=True)
    save_statistics(f'logs/experiments/{test_name}', statistics)


if __name__ == '__main__':
    all_data = load_dataset('src/data/ag_news.py')

    run_baseline({'warmup': 1, 'finetune': 5})
    run_ag_news_experiments({'warmup': 1, 'elmo': 1, 'finetune': 5})
