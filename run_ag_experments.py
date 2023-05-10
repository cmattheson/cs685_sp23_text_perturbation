from src.training.training import *
from src.models.bert_models import *
from datasets import load_dataset
from src.data.datasets import *
from torch.utils.data import DataLoader
from src.util import *
from transformers import BertModel

def run_baseline(phases):
    test_name = 'ag_news_bert_baseline'
    model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, num_classes))

    train_set = PerturbedSequenceDataset(all_data['train']['text'],
                                         torch.tensor(all_data['train']['label']), require_elmo_ids=False, perturb_characters=False, word_perturbation_rate=0.0)
    train_set, val_set = train_val_test_split(train_set, pct_train=0.8, pct_val=0.2)
    test_set = PerturbedSequenceDataset(all_data['test']['text'], torch.tensor(all_data['test']['label']),
                                        require_elmo_ids=False, perturb_characters=False, word_perturbation_rate=0.0)

    criterion = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    train_loader = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    statistics = train(model, train_loader, criterion, optim, device='cuda',
                       val_loader=val_loader,
                       test_loader=test_loader,
                       phases=phases,
                       record_training_statistics=True,
                       model_save_path=f'src/models/pretrained/{test_name}_model.pt',
                       record_time_statistics=True)
    save_statistics(f'logs/experiments/{test_name}', statistics)
    # ------------------------------------------------------------------------------------------------------------
    test_name = 'ag_news_bert_perturbed'
    model = ClassifierModel(BertModel.from_pretrained('bert-base-uncased'), nn.Linear(768, num_classes))

    train_set = PerturbedSequenceDataset(all_data['train']['text'],
                                         torch.tensor(all_data['train']['label']), require_elmo_ids=False)
    train_set, val_set = train_val_test_split(train_set, pct_train=0.8, pct_val=0.2)
    test_set = PerturbedSequenceDataset(all_data['test']['text'], torch.tensor(all_data['test']['label']),
                                        require_elmo_ids=False)

    criterion = torch.nn.functional.cross_entropy
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)
    train_loader = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=32, num_workers=2, shuffle=True, persistent_workers=True)
    statistics = train(model, train_loader, criterion, optim, device='cuda',
                       val_loader=val_loader,
                       test_loader=test_loader,
                       phases=phases,
                       record_training_statistics=True,
                       model_save_path=f'src/models/pretrained/{test_name}_model.pt',
                       record_time_statistics=True)
    save_statistics(f'logs/experiments/{test_name}', statistics)

def run_ag_news_experiments(phases):

    test_name = 'ag_news_bert_concat'
    model = ClassifierModel(Bert_Plus_Elmo_Concat(), nn.Linear(768, num_classes))

    train_set = PerturbedSequenceDataset(all_data['train']['text'],
                                         torch.tensor(all_data['train']['label']), require_elmo_ids=True)
    train_set, val_set = train_val_test_split(train_set, pct_train=0.8, pct_val=0.2)
    test_set = PerturbedSequenceDataset(all_data['test']['text'], torch.tensor(all_data['test']['label']),
                                        require_elmo_ids=True)

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
    num_classes = len(set(all_data['train']['label']))

    run_baseline({'warmup': 5, 'finetune': 10})
    run_ag_news_experiments({'warmup': 5, 'elmo': 5, 'finetune': 10})
