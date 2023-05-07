from typing import Callable

import torch
import torch.nn as nn
from tqdm import tqdm
from allennlp.modules.elmo import batch_to_ids
from transformers import BertTokenizer

def prepare_data(data, tokenizer, require_elmo_embeddings=True, device='cuda'):
    sequences, labels = data
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
                       tokenizer: BertTokenizer,
                       require_elmo_embeddings: bool = True,
                       device: str ='cuda') \
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
    accuracy = 0
    for i, data in enumerate(pbar):
        model_kwargs, labels = prepare_data(data,
                                            tokenizer,
                                            require_elmo_embeddings=require_elmo_embeddings,
                                            device=device)
        model.eval()
        with torch.no_grad():
            out = model(**model_kwargs)
        it_loss =  criterion(out.squeeze(), labels.to(torch.float32), reduction='sum').item()
        loss += it_loss
        pbar.set_description(
            f'it: {i + 1} / {len(dataloader)} loss: {it_loss}')
    return loss / len(dataloader.dataset), accuracy / len(dataloader.dataset)

def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: Callable,
          optim: torch.optim.Optimizer,
          tokenizer: BertTokenizer,
          num_epochs: int = 1,
          device: str = 'cuda',
          require_elmo_embeddings: bool = True,
          record_training_statistics: bool = False,
          record_time_statistics: bool = False,
          save_model_parameters: bool = False,
          *args,
          **kwargs
          ) -> dict[str, list[float]] or None:
    """

    Args:
        device:
        require_elmo_embeddings:
        save_model_parameters:
        record_time_statistics:
        record_training_statistics: whether to record the training statistics
        model: the model
        train_loader: training DataLoader
        criterion: the objective function
        optim: the optimizer
        tokenizer: tokenizer to use
        num_epochs: number of epochs to train for

    Returns:

    """
    import time

    statistics = {'training_loss': [], 'validation_loss': [], 'test_loss': [],
                  'training_accuracy': [], 'validation_accuracy': [], 'test_accuracy': [],
                  'total_time': None, 'model_compute_time': None, 'preprocessing_time': None}


    model.train()
    model.to(device)
    time_start = time.time()
    time_train = 0
    time_eval = 0
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader)
        model.train()
        for i, data in enumerate(pbar): # iterate over the perturbed sequences
            optim.zero_grad()
            model_kwargs, labels = prepare_data(data,
                                        tokenizer,
                                        require_elmo_embeddings=require_elmo_embeddings,
                                        device=device)
            time_start_forward = time.time()
            out = model(**model_kwargs)

            loss = criterion(out.squeeze(), labels.to(torch.float32))
            loss.backward()
            if record_training_statistics:
                statistics['training_loss'].append(loss.item())
            time_finish_backward = time.time()
            time_train += time_finish_backward - time_start_forward
            pbar.set_description(
                f'it: {len(pbar) * epoch + i + 1} / {len(pbar) * num_epochs} epoch: {epoch + 1} / {num_epochs} loss: {loss.item()}')
            optim.step()

        time_eval_start = time.time()
        if 'val_loader' in kwargs:
            val_loss, val_accuracy = compute_statistics(model,
                                                        criterion,
                                                        kwargs['val_loader'],
                                                        tokenizer,
                                                        require_elmo_embeddings=require_elmo_embeddings, device=device)
            statistics['validation_loss'].append(val_loss)
            statistics['validation_accuracy'].append(val_accuracy)
        if 'test_loader' in kwargs:
            test_loss, test_accuracy = compute_statistics(model,
                                                          criterion,
                                                          kwargs['test_loader'],
                                                          tokenizer,
                                                          require_elmo_embeddings=require_elmo_embeddings,
                                                          device=device)
            statistics['test_loss'].append(test_loss)
            statistics['test_accuracy'].append(test_accuracy)
        time_eval_finish = time.time()
        time_eval += time_eval_finish - time_eval_start


    if save_model_parameters:
        """
        save the model parameters if required

        """
        torch.save(model.state_dict(), 'model_parameters.pt')



    time_finish = time.time()
    total_time = time_finish - time_start

    if record_time_statistics:
        statistics['total_time'] = total_time
        statistics['model_compute_time'] = time_train
        statistics['preprocessing_time'] = total_time - time_train - time_eval

    statistics = {k: v for k, v in statistics.items() if v}

    return statistics




if __name__=='__main__':
    from transformers import BertTokenizer
    from bert_models import Bert_Plus_Elmo_Concat
    from bert_models import BinaryClassifierModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    myModel = Bert_Plus_Elmo_Concat()

    optim = torch.optim.Adam(myModel.parameters(), lr=0.0001)
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    from datasets import PerturbedSequenceDataset

    data = [('I loved the movie Guardians of the Galaxy 3', 1),
            ('I hated the movie Guardians of the Galaxy 3', 0)] * 64

    dataset = PerturbedSequenceDataset(data, log_directory='../../logs/character_perturbation')
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)


    classifier_head = nn.Linear(768, 1)
    m = BinaryClassifierModel(encoder=myModel, classifier=classifier_head)

    statistics = train(m, dataloader, criterion, optim, tokenizer, 1, device='cuda',
          record_training_statistics=True,
          record_time_statistics=True,
          save_model_parameters=True, val_loader=dataloader)


    print(statistics)
