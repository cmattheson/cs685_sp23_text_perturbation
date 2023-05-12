import torch
from typing import Iterable, Callable

from torch.utils.data import DataLoader

from src.data.datasets import PerturbedSequenceDataset
from src.models.bert_models import ClassifierModel
from src.training.training import train_val_test_split, train
from src.util import save_statistics


def run_experiment(name: str, phases: dict[str, int],
                   model: ClassifierModel,
                   optim: torch.optim.Optimizer,
                   train_data: dict[str, Iterable[str]],
                   test_data: dict[str, Iterable[str]] = None,
                   split: float = 0.8,
                   criterion: Callable = torch.nn.functional.cross_entropy,
                   require_elmo_ids=False,
                   train_char_perturbation_rate: float = 0.0,
                   train_word_perturbation_rate: float = 0.0,
                   val_char_perturbation_rate: float = 0.0,
                   val_word_perturbation_rate: float = 0.0,
                   batch_size: int = 32,
                   num_workers: int = 2

                   ) -> None:
    """

    Args:
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
    train_set = PerturbedSequenceDataset(train_data['text'],
                                         torch.tensor(train_data['label']),
                                         require_elmo_ids=require_elmo_ids,
                                         train_char_perturbation_rate=train_char_perturbation_rate,
                                         train_word_perturbation_rate=train_word_perturbation_rate,
                                         val_char_perturbation_rate=val_char_perturbation_rate,
                                         val_word_perturbation_rate=val_word_perturbation_rate)
    val_loader = None
    test_loader = None

    if split < 1.0:
        # if we have any validation data, split the train set into train and validation
        train_set, val_set = train_val_test_split(train_set, pct_train=split, pct_val=1.0 - split)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                persistent_workers=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              persistent_workers=True)

    if test_data:
        # if we have any test data, set up the test set and data loader
        test_set = PerturbedSequenceDataset(test_data['text'],
                                            torch.tensor(test_data['label']),
                                            require_elmo_ids=require_elmo_ids,
                                            train_char_perturbation_rate=train_char_perturbation_rate,
                                            train_word_perturbation_rate=train_word_perturbation_rate,
                                            val_char_perturbation_rate=val_char_perturbation_rate,
                                            val_word_perturbation_rate=val_word_perturbation_rate)
        test_set.eval()

        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                 persistent_workers=True)
    statistics = train(model, train_loader, criterion, optim, device='cuda',
                       val_loader=val_loader,
                       test_loader=test_loader,
                       phases=phases,
                       record_training_statistics=True,
                       model_save_path=f'src/models/pretrained/{name}_model.pt',
                       record_time_statistics=True)

    save_statistics(f'logs/experiments/{name}', statistics)
