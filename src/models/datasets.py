import torch
from torch.utils.data import Dataset, DataLoader
from my_tokenizers import *
from src.character_perturbation.text_perturbation import *
class CustomDataset(Dataset):
    def __init__(self, vocab, prob=1.0, per_char=0.3):
        self.vocab = vocab
        self.labels = torch.tensor([i for i in range(len(vocab))])
        self.tokenizer = Tokenizer(vocab)
        self.prob = prob
        self.per_char = per_char

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        r = random.random()
        if self.prob < r:
            word = corrupt(self.vocab[idx], prob=self.prob, per_char=self.per_char)
            while len(word) > 29 or word == self.vocab[idx]:
                word = corrupt(self.vocab[idx], prob=self.prob, per_char=self.per_char)
            token = self.tokenizer(word)
        else:
            word = self.vocab[idx]
            token = self.tokenizer(self.vocab[idx])
        return word, token, self.labels[idx]


class PerturbedSequenceDataset(Dataset):
    def __init__(self, data, log_directory, perturbation_rate=1.0, perturbation_rate_per_char=0.15):
        self.handler = TextPerturbationHandler(log_directory=log_directory)
        self.data = data
        self.perturbation_rate = perturbation_rate
        self.perturbation_rate_per_char = perturbation_rate_per_char

    def __getitem__(self, idx):
        itm = self.data[idx]
        label = itm[1]
        text = self.handler.perturb_string(itm[0])
        return text, label

    def __len__(self):
        return len(self.data)

