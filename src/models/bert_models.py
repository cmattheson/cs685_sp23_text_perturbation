import torch
import torch.nn as nn
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from transformers import BertTokenizer, BertModel
from allennlp.modules.elmo import _ElmoCharacterEncoder



class Bert_Plus_Elmo(nn.Module):
    """
    This model uses BERT and ELMo embeddings to produce a single embedding for each token.
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 256)
        self.elmo_embedder = _ElmoCharacterEncoder(options_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                                                   weight_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
        """Project the 128-dimensional ELMo vectors to 768 dimensions to match BERT's embedding dimension"""
        self.elmo_projection = nn.Linear(128, 768)


    def forward(self, bert_input_ids, elmo_input_ids, attention_mask=None):
        input_shape = bert_input_ids.size()
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape)

        print(
            'bert_input_ids.shape: ', bert_input_ids.shape,
        )
        print(
            'elmo_input_ids.shape: ', elmo_input_ids.shape,
        )
        bert_embedding = self.bert.embeddings.word_embeddings(bert_input_ids)
        elmo_embedding = self.elmo_projection(self.elmo_embedder(elmo_input_ids)['token_embedding'])
        elmo_embedding[:, 1:, :] = 0 # zero out the embeddings for the BOS token
        elmo_embedding[:, -1, :] = 0 # zero out the embeddings for the EOS token
        print('bert_embedding.shape: ', bert_embedding.shape)
        print('elmo_embedding.shape: ', elmo_embedding.shape)
        print('elmo_embedding: ', elmo_embedding)
        embeddings = bert_embedding + elmo_embedding
        output_representations = self.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
        )
        return output_representations



class Bert_Plus_Elmo_Skip(nn.Module):
    def __init__(self, elmo_strength_out=1.0, learn_elmo_strength=False):
        super().__init__()
        self.elmo_strength_out = elmo_strength_out
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 256)
        if learn_elmo_strength:
            self.elmo_strength_out = nn.Parameter(torch.tensor(elmo_strength_out))
        else:
            self.elmo_strength_out = elmo_strength_out
        self.elmo_embedder = _ElmoCharacterEncoder(options_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                                                   weight_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
        """Project the 128-dimensional ELMo vectors to 768 dimensions to match BERT's embedding dimension"""
        self.elmo_projection = nn.Linear(128, 768)


    def forward(self, bert_input_ids, elmo_input_ids, attention_mask=None):
        input_shape = bert_input_ids.size()
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape)

        bert_embedding = self.bert.embeddings.word_embeddings(bert_input_ids)
        elmo_embedding = self.elmo_projection(self.elmo_embedder(elmo_input_ids)['token_embedding'])
        elmo_embedding[:, 1:, :] = 0 # zero out the embeddings for the BOS token
        elmo_embedding[:, -1, :] = 0 # zero out the embeddings for the EOS token

        embeddings = bert_embedding + self.elmo_strength_in * elmo_embedding
        output_representations = self.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
        ) + self.elmo_strength_out*elmo_embedding
        return output_representations

class BERT_Plus_Correction(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

