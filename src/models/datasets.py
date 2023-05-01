import torch
from torch.utils.data import Dataset, DataLoader
from my_tokenizers import *
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
        if self.prob > 0.0:
            word = corrupt(self.vocab[idx], prob=self.prob, per_char=self.per_char)
            while len(word) > 29 or word == self.vocab[idx]:
                word = corrupt(self.vocab[idx], prob=self.prob, per_char=self.per_char)
            token = self.tokenizer(word)
        else:
            word = self.vocab[idx]
            token = self.tokenizer(self.vocab[idx])
        return word, token, self.labels[idx]
