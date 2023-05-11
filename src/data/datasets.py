import torch
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import Dataset
from transformers import BertTokenizer

from tokenizers import *
from src.character_perturbation.text_perturbation import *
from torch.nn.functional import pad
from src.word_perturbation.synonym_replacement import SynonymReplacementHandler


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
    """"
    """

    def __init__(self,
                 data,
                 labels,
                 log_directory='./logs/character_perturbation',
                 train_word_perturbation_rate=0.0,
                 val_word_perturbation_rate=0.0,
                 train_char_perturbation_rate=0.0,
                 val_char_perturbation_rate=0.0,
                 tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                 require_elmo_ids=True):
        self.train_char_perturbation_handler = TextPerturbationHandler(log_directory=log_directory,
                                                                       perturbation_weight=train_char_perturbation_rate)
        self.val_char_perturbation_handler = TextPerturbationHandler(log_directory=log_directory,
                                                                     perturbation_weight=val_char_perturbation_rate)
        self.train_synonym_replacement_handler = SynonymReplacementHandler(
            perturbation_chance=train_word_perturbation_rate)
        self.val_synonym_replacement_handler = SynonymReplacementHandler(perturbation_chance=val_word_perturbation_rate)
        self.data = data
        self.labels = labels
        self.train_char_perturbation_rate = train_char_perturbation_rate
        self.val_char_perturbation_rate = val_char_perturbation_rate
        self.train_word_perturbation_rate = train_word_perturbation_rate
        self.val_word_perturbation_rate = val_word_perturbation_rate
        self.train_word_perturbation_rate = train_word_perturbation_rate
        self.tokenizer = tokenizer
        self.require_elmo_ids = require_elmo_ids
        self.max_word_length = 50
        self.max_sentence_length = 50  # restrict to 50 for now so we don't run out of vram :(
        self.eval_mode = False

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.data[idx]

        if not self.eval_mode:
            if self.train_word_perturbation_rate > 0:
                text = self.train_synonym_replacement_handler.sentence_perturbe(text)
            if self.train_char_perturbation_rate > 0:
                text = self.train_char_perturbation_handler.perturb_string(text)
        else:
            if self.val_word_perturbation_rate > 0:
                text = self.val_synonym_replacement_handler.sentence_perturbe(text)
            if self.val_char_perturbation_rate > 0:
                text = self.val_char_perturbation_handler.perturb_string(text)
        if self.tokenizer:
            assert isinstance(text, str), f'Expected string, got {type(text)}'
            encoded_input = self.tokenizer(text, return_tensors='pt')
            input_ids = encoded_input['input_ids'].squeeze(0)
            input_ids = pad(input_ids, (0, self.max_sentence_length - input_ids.shape[0]), value=0)
            attention_mask = encoded_input['attention_mask'].squeeze(0)
            attention_mask = pad(attention_mask, (0, self.max_sentence_length - attention_mask.shape[0]), value=0)
            if self.require_elmo_ids:
                elmo_input_ids = batch_to_ids([self.tokenizer.tokenize(text)]).squeeze(0)
                elmo_input_ids = pad(elmo_input_ids, (0, self.max_word_length - elmo_input_ids.shape[1], 0,
                                                      self.max_sentence_length - elmo_input_ids.shape[0]), value=0)

                return input_ids, elmo_input_ids, attention_mask, label
            else:
                return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data)

    def train(self):
        self.eval_mode = False

    def eval(self):
        self.eval_mode = True
