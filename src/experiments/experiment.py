import os

import torch
from typing import Iterable, Callable

from torch.utils.data import DataLoader

from src.data.datasets import PerturbedSequenceDataset
from src.models.bert_models import ClassifierModel
from src.training.training import train_val_test_split, train
from src.util import save_statistics

from datasets.arrow_dataset import Dataset as ArrowDataset

def run_experiment(name: str, phases: dict[str, int],
                   model: ClassifierModel,
                   optim: torch.optim.Optimizer,
                   train_data: ArrowDataset or dict[str, Iterable[str]] or PerturbedSequenceDataset = None,
                   val_data: ArrowDataset or dict[str, Iterable[str]] or PerturbedSequenceDataset = None,
                   test_data: ArrowDataset or dict[str, Iterable[str]] or PerturbedSequenceDataset = None,
                   split: float = 0.8,
                   criterion: Callable = torch.nn.functional.cross_entropy,
                   require_elmo_ids=False,
                   train_char_perturbation_rate: float = 0.0,
                   train_word_perturbation_rate: float = 0.0,
                   val_char_perturbation_rate: float = 0.0,
                   val_word_perturbation_rate: float = 0.0,
                   batch_size: int = 32,
                   num_workers: int = 2,
                   save_datasets=True

                   ) -> None:
    """

    Args:
        val_data: the valadation data
        criterion: the criterion from nn.functional to use
        optim: the optimizer for the model
        batch_size: batch size
        num_workers: num workers for dataloader
        name: the name of the current experiment
        phases: the phases in training
        model: the model to train
        train_data: the training data (will be split into train and validation)
        test_data: the test data
        split: how much data to use for training (the rest is used for validation) (0.8 means 80% of the data is used for training)
        require_elmo_ids: whether to require elmo ids (needed for the models incorporating elmo encodings)
        train_char_perturbation_rate: how much char perturbation to use for training
        train_word_perturbation_rate: how much word perturbation to use for training
        val_char_perturbation_rate: how much char perturbation to use for validation and testing
        val_word_perturbation_rate: how much word perturbation to use for validation and testing

    Returns:

    """

    # train set with no perturbation
    print(f'running experiment with train_char_perturbation_rate={train_char_perturbation_rate}, '
          f'val_char_perturbation_rate={val_char_perturbation_rate}, '
          f'train_word_perturbation_rate={train_word_perturbation_rate}, '
          f'val_word_perturbation_rate={val_word_perturbation_rate}')

    val_set = None
    test_set = None
    if os.path.isfile(f'src/data/datasets/{name}_test.pt'):
        print('Loading test set from file')
        test_set = torch.load(f'src/data/datasets/{name}_test.pt')
        test_set.eval()
    elif isinstance(test_data, PerturbedSequenceDataset):
        print('using test set from args')
        test_set = test_data
        test_set.eval()
    elif test_data:
        print('Creating test set from dict')
        test_set = PerturbedSequenceDataset(test_data['text'], test_data['label'], require_elmo_ids=require_elmo_ids,
                                            train_char_perturbation_rate=val_char_perturbation_rate,
                                            train_word_perturbation_rate=val_word_perturbation_rate,
                                            val_char_perturbation_rate=val_char_perturbation_rate,
                                            val_word_perturbation_rate=val_word_perturbation_rate)
        test_set.eval()


    print('Creating datasets')  # create the datasets
    if isinstance(train_data, ArrowDataset) or isinstance(train_data, dict):
        train_set = PerturbedSequenceDataset(train_data['text'],
                                             torch.tensor(train_data['label']),
                                             require_elmo_ids=require_elmo_ids,
                                             train_char_perturbation_rate=train_char_perturbation_rate,
                                             train_word_perturbation_rate=train_word_perturbation_rate,
                                             val_char_perturbation_rate=val_char_perturbation_rate,
                                             val_word_perturbation_rate=val_word_perturbation_rate)
    else:
        print('using train set from args')
        train_set = train_data

    if val_data:
        if isinstance(val_data, ArrowDataset) or isinstance(val_data, dict):
            val_set = PerturbedSequenceDataset(val_data['text'],
                                               torch.tensor(val_data['label']),
                                               require_elmo_ids=require_elmo_ids,
                                               train_char_perturbation_rate=train_char_perturbation_rate,
                                               train_word_perturbation_rate=train_word_perturbation_rate,
                                               val_char_perturbation_rate=val_char_perturbation_rate,
                                               val_word_perturbation_rate=val_word_perturbation_rate)
        else:
            print('using val set from args')
            val_set = val_data
    elif split < 1.0:
        # if we have any validation data, split the train set into train and validation
        train_set, val_set = train_val_test_split(train_set, pct_train=split, pct_val=1.0 - split)

    # save the created datasets
    if save_datasets:
        print('Saving datasets')
        os.makedirs(f'src/data/datasets/', exist_ok=True)
        if not os.path.isfile(f'src/data/datasets/{name}_train.pt'):
            torch.save(train_set, f'src/data/datasets/{name}_train.pt')
        if not os.path.isfile(f'src/data/datasets/{name}_val.pt'):
            torch.save(val_set, f'src/data/datasets/{name}_val.pt')
        if test_set:
            if not os.path.isfile(f'src/data/datasets/{name}_test.pt'):
                torch.save(test_set, f'src/data/datasets/{name}_test.pt')
    # end of dataset creation

    if val_set:
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                persistent_workers=True, prefetch_factor=2)
    else:
        val_loader = None
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              persistent_workers=True, prefetch_factor=2)

    test_loader = None

    if test_set:
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                 persistent_workers=True, prefetch_factor=2)
    # train the model and save the statistics
    statistics = train(model, train_loader, criterion, optim, device='cuda',
                       val_loader=val_loader,
                       test_loader=test_loader,
                       phases=phases,
                       record_training_statistics=True,
                       model_save_path=f'src/models/pretrained/{name}/',
                       record_time_statistics=True)

    save_statistics(f'logs/experiments/{name}', statistics)
