import torch
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import Dataset
from transformers import BertTokenizer

from tokenizers import *
from src.character_perturbation.perturbation_handler import *
from torch.nn.functional import pad
from src.word_perturbation.perturbation_handler import perturb_text_with_synonyms


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
                 word_perturbation_rate=0.0,
                 perturb_characters=True,
                 tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                 require_elmo_ids=True):
        self.handler = CharacterPerturbationHandler(log_directory=log_directory)
        self.data = data
        self.labels = labels
        self.perturb_characters = perturb_characters
        self.word_perturbation_rate = word_perturbation_rate
        self.tokenizer = tokenizer
        self.require_elmo_ids = require_elmo_ids
        self.max_word_length = 50
        self.max_sentence_length = 50  # restrict to 50 for now so we don't run out of vram :(

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.data[idx]

        if self.word_perturbation_rate > 0:
            text = perturb_text_with_synonyms(text, self.word_perturbation_rate)
        if self.perturb_characters:
            text = self.handler.perturb_string(text)
        if self.tokenizer:
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


class PerturbedSequenceDataset(Dataset):
    """"
    """

    def __init__(self,
                 data,
                 labels,
                 log_directory='./logs/character_perturbation',
                 word_perturbation_rate=0.0,
                 train_char_perturbation_rate=0.0,
                 val_char_perturbation_rate=5.0,
                 tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'),
                 require_elmo_ids=True):
        self.handler_train = CharacterPerturbationHandler(log_directory=log_directory,
                                                     perturbation_weight=train_char_perturbation_rate)
        self.handler_val = CharacterPerturbationHandler(log_directory=log_directory,
                                                   perturbation_weight=val_char_perturbation_rate)
        self.data = data
        self.labels = labels
        self.char_perturbation_rate = train_char_perturbation_rate
        self.val_char_perturbation_rate = val_char_perturbation_rate
        self.word_perturbation_rate = word_perturbation_rate
        self.tokenizer = tokenizer
        self.require_elmo_ids = require_elmo_ids
        self.max_word_length = 50
        self.max_sentence_length = 50  # restrict to 50 for now so we don't run out of vram :(
        self.eval_mode = False

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.data[idx]
        if self.word_perturbation_rate > 0:
            text = perturb_text_with_synonyms(text, self.word_perturbation_rate)
        if not self.eval_mode:
            if self.char_perturbation_rate > 0:
                text = self.handler_train.perturb_string(text)
        else:
            if self.val_char_perturbation_rate > 0:
                text = self.handler_val.perturb_string(text)
        if self.tokenizer:
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
