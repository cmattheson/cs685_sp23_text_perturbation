import torch
import torch.nn as nn
from torch.nn.functional import relu
from allennlp.modules.transformer.positional_encoding import SinusoidalPositionalEncoding


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_word_length):
        super().__init__()
        self.max_word_length = max_word_length
        self.char_embedding = nn.Embedding(28, 8)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos1 = nn.Linear(1, 8)
        self.pos2 = nn.Linear(8, 8)
        self.pos3 = nn.Linear(8, 8)
        self.chartoword_embedding = nn.Linear(8 * max_word_length, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    def forward(self, charlevel_tokens):
        char_embed = self.char_embedding(charlevel_tokens)
        positions = torch.arange(0, self.max_word_length, dtype=torch.float32).reshape(-1, 1).to(
            charlevel_tokens.device)
        pos = relu(self.pos1(positions))
        pos = relu(self.pos2(pos))
        pos = self.pos3(pos)
        x = char_embed + pos
        x = x.flatten(start_dim=1)
        x = relu(self.chartoword_embedding(x))
        x = relu(self.linear1(x))
        return self.classifier(x)


class NetNoPosEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_word_length):
        super().__init__()
        self.max_word_length = max_word_length
        self.char_embedding = nn.Embedding(28, 8)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.chartoword_embedding = nn.Linear(8 * max_word_length, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    def forward(self, charlevel_tokens):
        char_embed = self.char_embedding(charlevel_tokens)
        x = char_embed
        x = x.flatten(start_dim=1)
        x = relu(self.chartoword_embedding(x))
        x = relu(self.linear1(x))
        return self.classifier(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_word_length=30, nhead=8, num_encoder_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding()
        self.max_word_length = max_word_length
        self.char_embedding = nn.Embedding(28, d_model)
        # self.chartoword_embedding = nn.Linear(8 * max_word_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        char_embed = self.char_embedding(tokens)
        x = char_embed
        x = self.pos(x)
        # x = x.flatten(start_dim=1)
        # x = relu(self.chartoword_embedding(x))
        x = x[:, 0, :]
        x = self.transformer_encoder(x)
        return self.classifier(x)
