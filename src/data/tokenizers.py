import torch
import random

from transformers import BertTokenizer
from allennlp.modules.elmo import batch_to_ids


test_seqs = ['this is a test', 'this is another test']
class LMSpellcheckderTokenizer:
    def __init__(self, vocab):
        characters = [*'abcdefghijklmnopqrstuvwxyz']
        self.char_tokens = dict()
        self.word_tokens = dict()
        self.labels = dict()
        self.max_word_length = 0
        self.vocab = vocab

        self.char_tokens['<pad>'] = 0
        self.char_tokens['<cls>'] = 27
        for i, char in enumerate(characters):
            self.char_tokens[char] = i + 1

    def tokenize(self, x: str or list):
        if isinstance(x, str):
            words = [x]
        else:
            words = x
        sequences = []
        for word in words:
            ctokens = [self.char_tokens['<cls>']]
            for char in word:
                ctokens.append(self.char_tokens[char])
            while len(ctokens) < 30:
                ctokens.append(self.char_tokens['<pad>'])
            sequences.append(ctokens)

        tokens = torch.tensor(sequences).squeeze()
        return tokens

    def attention_mask(self, n):
        x = torch.arange(0, n)
        y = x.view(-1, 1)
        mask = torch.where(x <= y, 1, 0)
class Tokenizer:
    def __init__(self, vocab):
        characters = [*'abcdefghijklmnopqrstuvwxyz']
        self.char_tokens = dict()
        self.word_tokens = dict()
        self.labels = dict()
        self.num_classes = len(vocab)
        self.max_word_length = 0
        self.vocab = vocab


        self.char_tokens['<pad>'] = 0
        self.char_tokens['<cls>'] = 27
        for i, char in enumerate(characters):
            self.char_tokens[char] = i + 1
        self.word_tokens['<unk>'] = 0
        self.word_tokens['<pad>'] = 1
        for i, word in enumerate(vocab):
            self.word_tokens[word] = i + 2
            self.labels[word] = i

    def __call__(self, words):
        return self.tokenize_chars(words)
    def tokenize_chars(self, word):
        if isinstance(word, str):
            words = [word]
        else:
            words = word
        sequences = []
        for word in words:
            ctokens = [self.char_tokens['<cls>']]
            for char in word:
                ctokens.append(self.char_tokens[char])
            while len(ctokens) < 30:
                ctokens.append(self.char_tokens['<pad>'])
            sequences.append(ctokens)

        tokens = torch.tensor(sequences).squeeze()
        return tokens
    def detokenize(self, tokenized_words):
        skip = set(self.char_tokens['CLS'], self.char_tokens['PAD'])
        words = []
        for token in tokenized_words:
            word = []
            for id in token:
                if id in skip:
                    continue
                word.append(self.char_tokens[id])
            words.append(''.join(word))
        return words

    def tokenize_words(self, words):
        wtokens = []
        for i, word in enumerate(words):
            if word in self.word_tokens:
                wtokens.append(self.word_tokens[word])
            else:
                wtokens.append(self.word_tokens['<unk>'])
        data = torch.tensor(wtokens)
        labels = torch.tensor([self.labels[word] for word in words])
        return data, labels

def corrupt(word, prob=1, per_char=0.1):
    alphabet = [*'abcdefghijklmnopqrstuvwxyz']
    letters = []
    # replace a letter with 5% probability
    # delete a letter with 5% probability
    # duplicate a letter with 5% probability
    # insert a letter with 5% probability
    # with small probability, return the uncorrupted word
    if random.random() > prob:
        return word

    for char in word:
        roll = random.random()

        if roll < 0.2*per_char:
            # replace char
            letters.append(alphabet[int(random.random() * len(alphabet))])
        elif roll < 0.4*per_char:
            # delete char
            continue
        elif roll < 0.6*per_char:
            # duplicate char
            letters.append(char)
            letters.append(char)
        elif roll < 0.8*per_char:
            # append char
            letters.append(alphabet[int(random.random() * len(alphabet))])
            letters.append(char)
        elif roll < per_char:
            # prepend char
            letters.append(char)
            letters.append(alphabet[int(random.random() * len(alphabet))])
        else:
            # do nothing
            letters.append(char)
    return ''.join(letters)


