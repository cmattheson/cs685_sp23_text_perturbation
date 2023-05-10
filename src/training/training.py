from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from datasets import load_dataset

from src.models.bert_models import ElmoBertModel


def train_val_test_split(dataset: torch.utils.data.Dataset,
                         pct_train=0.8,
                         pct_val=0.2) \
        -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset] or tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """

    Args:
        dataset:

    Returns:

    """
    from torch.utils.data import Dataset
    train_size = int(pct_train * len(dataset))
    val_size = int(pct_val * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    if pct_train + pct_val == 1:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    else:
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train_dataset, val_dataset, test_dataset


def prepare_data(data: tuple[tuple[str], torch.tensor],
                 tokenizer: BertTokenizer,
                 require_elmo_embeddings: bool = True,
                 device: str = 'cuda'):
    """

    Args:
        data:
        tokenizer:
        require_elmo_embeddings:
        device:

    Returns:

    """
    sequences, labels = data
    print(type(sequences[0]))
    labels = labels.to(device)
    encoded_input = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)

    bert_input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    if require_elmo_embeddings:
        elmo_input_ids = batch_to_ids([tokenizer.tokenize(sentence) for sentence in sequences]).to(device)
        model_kwargs = {'input_ids': bert_input_ids, 'elmo_input_ids': elmo_input_ids, 'attention_mask': attention_mask}
    else:
        model_kwargs = {'input_ids': bert_input_ids, 'attention_mask': attention_mask}
    return model_kwargs, labels


def compute_statistics(model: nn.Module,
                       criterion: Callable,
                       dataloader: torch.utils.data.DataLoader,
                       device: str = 'cuda') \
        -> tuple[float, float]:
    """

    Args:
        model: nn.
        criterion:
        dataloader:
        tokenizer:
        require_elmo_embeddings:
        device:

    Returns: (loss, accuracy)

    """
    pbar = tqdm(dataloader)
    loss = 0
    num_correct = 0
    num_examples = 0
    for i, data in enumerate(pbar):
        """
        model_kwargs, labels = prepare_data(data,
                                            tokenizer,
                                            require_elmo_embeddings=require_elmo_embeddings,
                                            device=device)
        """
        model.eval()

        if isinstance(model.encoder, ElmoBertModel):
            input_ids, elmo_input_ids, attention_mask, labels = data[0].to(device), \
                data[1].to(device), data[2].to(device), data[3].to(device)
            with torch.no_grad():
                out = model(input_ids, elmo_input_ids, attention_mask)
        else:
            input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(
                device)
            with torch.no_grad():
                out = model(input_ids, attention_mask)

        it_loss = criterion(out.squeeze(), labels, reduction='sum').item()
        # check if we are doing binary classification or multiclass
        if out.shape[1] == 1:
            it_num_correct = torch.sum(torch.round(torch.sigmoid(out.squeeze())) == labels).item()
        else:
            it_num_correct = torch.sum(torch.argmax(out, dim=1) == labels).item()
        num_correct += it_num_correct
        loss += it_loss
        pbar.set_description(
            f'it: {i + 1} / {len(dataloader)} loss: {it_loss / len(data[0])}')

    accuracy = num_correct / len(dataloader.dataset)
    return loss / len(dataloader.dataset), accuracy


def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: Callable,
          optim: torch.optim.Optimizer,
          *,
          num_epochs: int = None,
          device: str = 'cuda',
          record_training_statistics: bool = False,
          record_time_statistics: bool = False,
          model_save_path: str = None,
          phases: dict[str, int] = None,
          **kwargs
          ) -> dict[str, list[float]] or None:
    """

    Args:
        phases: the phases, a dict where the key is the phase name and the value is the number of epochs to train for
        device: the device to train on, default 'cuda'
        save_model_parameters: whether to save the model parameters
        record_time_statistics: whether to record the time statistics
        record_training_statistics: whether to record the training statistics
        model: the model to train
        train_loader: training DataLoader
        criterion: the objective function
        optim: the optimizer
        tokenizer: tokenizer to use
        num_epochs: number of epochs to train for

    Returns:

    """
    assert not phases or not num_epochs, 'Either specify the number of epochs or the phases'
    import time

    statistics = {'training_loss': [], 'validation_loss': [], 'test_loss': [],
                  'training_accuracy': [], 'validation_accuracy': [], 'test_accuracy': [],
                  'total_time': None, 'model_compute_time': None, 'preprocessing_time': None}
    if isinstance(model.encoder, ElmoBertModel):
        print('Using ElmoBertModel')
        require_elmo_embeddings: bool = True
    else:
        require_elmo_embeddings: bool = False
        print('Using BertModel')
    model.train()
    model.to(device)
    time_start = time.time()
    time_train = 0
    time_eval = 0
    if not phases and not num_epochs:
        num_epochs = 1
        phases = {'finetune': num_epochs}
    for phase, num_epochs in phases.items():
        print(f'Phase: {phase}')
        model.setPhase(phase)
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader)
            model.train()
            for i, data in enumerate(pbar):  # iterate over the perturbed sequences
                optim.zero_grad()
                if require_elmo_embeddings:
                    input_ids, elmo_input_ids, attention_mask, labels = data[0].to(device), \
                        data[1].to(device), data[2].to(device), data[3].to(device)
                    time_start_forward = time.time()
                    out = model(input_ids, elmo_input_ids, attention_mask=attention_mask)
                else:
                    input_ids, attention_mask, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                    time_start_forward = time.time()
                    out = model(input_ids, attention_mask)
                loss = criterion(out.squeeze(), labels)
                loss.backward()
                time_finish_backward = time.time()
                time_train += time_finish_backward - time_start_forward

                if record_training_statistics:
                    statistics['training_loss'].append(loss.item())
                    # check if we are doing binary or multiclass classification
                    if out.shape[1] == 1:
                        statistics['training_accuracy'].append(
                            torch.sum(torch.round(torch.sigmoid(out.squeeze())) == labels).item() / len(labels))
                    else:
                        statistics['training_accuracy'].append(
                            torch.sum(torch.argmax(out.squeeze(), dim=1) == labels).item() / len(labels))

                pbar.set_description(
                    f'it: {len(pbar) * epoch + i + 1} / {len(pbar) * num_epochs} epoch: {epoch + 1} / {num_epochs} loss: {loss.item()}')
                optim.step()

            time_eval_start = time.time()

            if 'val_loader' in kwargs:
                print('Evaluating on validation set')
                val_loss, val_accuracy = compute_statistics(model,
                                                            criterion,
                                                            kwargs['val_loader'],
                                                            device=device)
                statistics['validation_loss'].append(val_loss)
                statistics['validation_accuracy'].append(val_accuracy)
            if 'test_loader' in kwargs:
                print('Evaluating on test set')
                test_loss, test_accuracy = compute_statistics(model,
                                                              criterion,
                                                              kwargs['test_loader'],
                                                              device=device)
                statistics['test_loss'].append(test_loss)
                statistics['test_accuracy'].append(test_accuracy)
            time_eval_finish = time.time()
            time_eval += time_eval_finish - time_eval_start

    # record the total time to finish all operations
    time_finish = time.time()
    total_time = time_finish - time_start

    if model_save_path:
        """
        save the model parameters if required

        """
        torch.save(model.state_dict(), model_save_path)

    if record_time_statistics:
        statistics['total_time'] = total_time
        statistics['model_compute_time'] = time_train
        statistics['preprocessing_time'] = total_time - time_train - time_eval

    statistics = {k: v for k, v in statistics.items() if v}

    return statistics


if __name__ == '__main__':
    dataset = load_dataset('../data/ag_news.py')

    num_classes = len(set(dataset['train']['label']))

    from transformers import BertTokenizer
    from src.models.bert_models import Bert_Plus_Elmo_Concat, Bert_Plus_Elmo, ClassifierModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoder = Bert_Plus_Elmo_Concat(options_file='../models/pretrained/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                                    weight_file='../models/pretrained/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
   # encoder = Bert_Plus_Elmo(options_file='../models/pretrained/elmo_2x1024_128_2048cnn_1xhighway_options.json',
   #                          weight_file='../models/pretrained/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

    # optim = torch.optim.Adam(encoder.parameters(), lr=0.0003)

    # criterion = torch.nn.functional.binary_cross_entropy_with_logits
    criterion = torch.nn.functional.cross_entropy

    # examples = [('I loved the movie Guardians of the Galaxy 3', 1),
    #            ('I hated the movie Guardians of the Galaxy 3', 0)] * 1000

    # data = [itm[0] for itm in examples]
    # labels = torch.tensor([itm[1] for itm in examples])

    # dataset = PerturbedSequenceDataset2(data, tokenizer=tokenizer, log_directory='../../logs/character_perturbation')
    from torch.utils.data import DataLoader

    # dataloader = DataLoader(dataset, batch_size=64, num_workers=1, shuffle=True)
    from src.data.datasets import *

    dataset = PerturbedSequenceDataset(dataset['train']['text'][:1000], torch.tensor(dataset['train']['label'][:1000]), log_directory='../../logs/character_perturbation')
    train_set, val_set = train_val_test_split(dataset)
    train_loader = DataLoader(train_set, batch_size=64, num_workers=2, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2, shuffle=True, persistent_workers=True)
    print(type(train))
    # create a binary classifier head
    classifier_head = nn.Linear(768, num_classes)
    model = ClassifierModel(encoder=encoder, classifier=classifier_head)
    phases = {'finetune': 1}
    optim = torch.optim.Adam(model.parameters(), lr=0.00003)

    statistics = train(model, train_loader, criterion, optim, device='cuda',
                       val_loader=val_loader,
                       phases=phases,
                       record_training_statistics=True,
                       model_save_path='../models/pretrained/bert_elmo_ag_news_classifier.pt',
                       record_time_statistics=True)

    print(statistics)

    test_name = 'ag_news_'
    for stat in statistics:
        with open(test_name + f'{stat}.txt', 'w') as f:
            f.write(str(statistics[stat]))
