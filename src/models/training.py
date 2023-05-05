import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from allennlp.modules.elmo import batch_to_ids


def train(model, dataloader, criterion, optim, tokenizer, num_epochs=1, device='cuda', require_elmo_embeddings=True):
    """

    Args:
        model: the model
        dataloader: dataloader
        criterion: the objective
        optim: the optimizer
        tokenizer: tokenizer to use
        num_epochs: number of epochs to train for

    Returns:

    """
    import time

    model.train()
    model.to(device)
    time_start = time.time()
    time_train = 0
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)
        for i, (sequences, labels) in enumerate(pbar): # iterate over the perturbed sequences
            labels = labels.to(device)
            optim.zero_grad()
            encoded_input = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)

            bert_input_ids = encoded_input['input_ids'].to(device)
            attention_mask = encoded_input['attention_mask'].to(device)
            if require_elmo_embeddings:
                elmo_input_ids = batch_to_ids([tokenizer.tokenize(sentence) for sentence in sequences]).to(device)
                kwargs = {'input_ids': bert_input_ids, 'elmo_input_ids': elmo_input_ids, 'attention_mask': attention_mask}
            else:
                kwargs = {'input_ids': bert_input_ids, 'attention_mask': attention_mask}
            time_start_forward = time.time()
            out = model(**kwargs)

            loss = criterion(out.squeeze(), labels.to(torch.float32))
            loss.backward()
            time_finish_backward = time.time()
            time_train += time_finish_backward - time_start_forward
            pbar.set_description(
                f'it: {len(pbar) * epoch + i + 1} / {len(pbar) * num_epochs} epoch: {epoch + 1} / {num_epochs} loss: {loss.item()}')
            optim.step()
    time_finish = time.time()
    total_time = time_finish - time_start
    percent_train = 100 * time_train / total_time
    print('total time: ', total_time)
    print('time forward: ', time_train)
    print('percent forward: ', percent_train)


def main():
    import torch
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    from bert_models import Bert_Plus_Elmo_Concat

    myModel = Bert_Plus_Elmo_Concat()

    optim = torch.optim.Adam(myModel.parameters(), lr=0.0001)
    criterion = torch.nn.functional.binary_cross_entropy_with_logits

    from datasets import PerturbedSequenceDataset
    data = [('I loved the movie Guardians of the Galaxy 3', 1), ('I hated the movie Guardians of the Galaxy 3', 0)] * 2000

    dataset = PerturbedSequenceDataset(data, log_directory='../../logs/character_perturbation')
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)

    from bert_models import BinaryClassifierModel
    import torch.nn as nn
    classifier_head = nn.Linear(768, 1)
    m = BinaryClassifierModel(encoder=myModel, classifier=classifier_head)

    train(m, dataloader, criterion, optim, tokenizer, 10, device='cuda')

if __name__=='__main__':
    main()