import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from allennlp.modules.elmo import _ElmoCharacterEncoder, batch_to_ids

class ElmoBertModel(nn.Module):
    def __init__(self):
        super().__init__()
class Bert_Plus_Elmo(ElmoBertModel):
    """
    This model sums the BERT and ELMo embeddings
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.elmo_embedder = _ElmoCharacterEncoder(options_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                                                   weight_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
        # Project the 128-dimensional ELMo vectors to 768 dimensions to match BERT's embedding dimension
        self.elmo_projection = nn.Linear(128, 768)




    def forward(self, input_ids, elmo_input_ids, attention_mask=None):
        seq_len = input_ids.shape[1]
        position_ids: torch.Tensor = self.bert.embeddings.position_ids[:, :seq_len]
        position_embedding: torch.Tensor = self.bert.embeddings.position_embeddings(position_ids)
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape)
        bert_embedding: torch.Tensor = self.bert.embeddings.word_embeddings(input_ids)
        elmo_embedding: torch.Tensor = self.elmo_projection(self.elmo_embedder(elmo_input_ids)['token_embedding'])
        embeddings: torch.Tensor = bert_embedding + elmo_embedding[:, 1:elmo_embedding.size(1)-1, :] + position_embedding
        embeddings: torch.Tensor = self.bert.embeddings.LayerNorm(embeddings)
        embeddings: torch.Tensor = self.bert.embeddings.dropout(embeddings)
        output_representations: torch.Tensor = self.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
        )
        return output_representations




class Bert_Plus_Elmo_Skip(ElmoBertModel):
    """
    This model sums the BERT and ELMo embeddings (not fully implemented)
    """
    def __init__(self, elmo_strength_out=1.0, learn_elmo_strength=False):
        super().__init__()
        self.elmo_strength_out = elmo_strength_out
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if learn_elmo_strength:
            self.elmo_strength_out = nn.Parameter(torch.tensor(elmo_strength_out))
        else:
            self.elmo_strength_out = elmo_strength_out
        self.elmo_embedder = _ElmoCharacterEncoder(
            options_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_options.json',
            weight_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        )
        # Project the 128-dimensional ELMo vectors to 768 dimensions to match BERT's embedding dimension
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


class Bert_Plus_Elmo_Concat(ElmoBertModel):
    """
    This model uses BERT and ELMo embeddings to produce a single embedding for each token. Unlike the additive model,
    it concatenates the BERT and ELMo embeddings along the sequence dimension. This should allow the model to attend to
    both the BERT and ELMo embeddings at the same time.
    """
    def __init__(self,  elmo_embedder_dim=128, elmo_strength=1.0, add_elmo_positional_encoding=True):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 256)
        self.elmo_embedder = _ElmoCharacterEncoder(options_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                                                   weight_file='pretrained/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
        # Project the 128-dimensional ELMo vectors to 768 dimensions to match BERT's embedding dimension
        self.elmo_projection = nn.Linear(elmo_embedder_dim, 768)
        self.elmo_strength = elmo_strength
        self.add_elmo_positional_encoding = add_elmo_positional_encoding


    def forward(self, input_ids, elmo_input_ids, attention_mask=None, layer_norm_elmo_separately=True):
        """

        Args:
            bert_input_ids: tensor of shape (batch_size, seq_len), where bert_seq_len is the length of the tokenized sequence
            elmo_input_ids: input to the ELMo embedder. This is a tensor of shape (batch_size, seq_len, 50)
            bert_attention_mask: attention mask provided by the BERT tokenizer
        Returns:

        """
        seq_len = input_ids.shape[1]
        position_ids = self.bert.embeddings.position_ids[:, :seq_len]
        input_embeds = self.bert.embeddings.word_embeddings(input_ids)
        position_embedding = self.bert.embeddings.position_embeddings(position_ids)
        bert_embedding = input_embeds + position_embedding
        if layer_norm_elmo_separately:
            bert_embedding = self.bert.embeddings.LayerNorm(bert_embedding)
            bert_embedding = self.bert.embeddings.dropout(bert_embedding)

        # get the elmo encoding
        #print(elmo_input_ids.shape)
        elmo_encoding = self.elmo_embedder(elmo_input_ids)
        #print('token embedding', elmo_encoding['token_embedding'].shape)
        # elmo_encoding['token_embedding'] Shape: (batch_size, seq_len, 128)
        elmo_embedding = self.elmo_projection(elmo_encoding['token_embedding'])  # shape (batch_size, seq_len, 768)

        if self.add_elmo_positional_encoding:
            # use the same positional embedding for elmo as for bert
            #print(position_embedding.shape)

            # drop the first and last position embeddings, since they are for the BOS and EOS tokens
            elmo_embedding = elmo_embedding[:, 1:elmo_embedding.size(1) - 1, :] + position_embedding
        if layer_norm_elmo_separately:
            elmo_embedding = self.bert.embeddings.LayerNorm(elmo_embedding)
            elmo_embedding = self.bert.embeddings.dropout(elmo_embedding)

        # get the attention mask for elmo (should be the same as the bert mask)
        elmo_attention_mask = elmo_encoding['mask']
        #print('elmo mask', elmo_attention_mask.shape)
        assert elmo_attention_mask is not None
        assert attention_mask is not None
        #print('bert mask', attention_mask.shape)

        # combine the bert and elmo embeddings by concatenating them along the sequence length dimension
        input_shape = (input_ids.size()[0], input_ids.size()[1] + elmo_embedding.shape[1])
        elmo_attention_mask.to(torch.int64)
        combined_attention_mask = torch.cat([attention_mask, attention_mask], dim=1)
        #print('combined attention mask', combined_attention_mask.shape)
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(combined_attention_mask, input_shape)
        #print('extended attention mask', extended_attention_mask.shape)
        combined_embedding = torch.concat([bert_embedding, elmo_embedding], dim=1)
        if not layer_norm_elmo_separately:
            combined_embedding = self.bert.embeddings.LayerNorm(combined_embedding)
            combined_embedding = self.bert.embeddings.dropout(combined_embedding)
        #  combined_embedding = bert_embedding
        #  extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask,
                                                                                      #input_ids.shape)
        output_representations = self.bert.encoder(
            combined_embedding,
            attention_mask=extended_attention_mask,
        )

        return output_representations

class BinaryClassifierModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.phases = {'warmup', 'finetune', 'elmo'}
        self.phase = 'finetune'

    def forward(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs: input_ids, elmo_input_ids, attention_mask, labels

        Returns:

        """
        x = self.encoder(*args, **kwargs)
        return self.classifier(x.last_hidden_state[:, 0])

    def setPhase(self, phase: str):
        assert phase in self.phases
        self.phase = phase
        if phase == 'warmup':
            """
            Freeze the encoder and only train the classifier
            """
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True
        elif phase == 'finetune':
            """
            Unfreeze the encoder and train both the encoder and the classifier
            """
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
        elif phase == 'elmo':
            """
            Freeze the bert encoder and only train the elmo encoder
            """
            assert isinstance(self.encoder, ElmoBertModel)
            for param in self.encoder.bert.parameters():
                param.requires_grad = False
        elif phase == 'freeeze_elmo':
            assert isinstance(self.encoder, ElmoBertModel)
            for param in self.encoder.elmo_embedder.parameters():
                param.requires_grad = False

if __name__=='__main__':
    model = Bert_Plus_Elmo_Concat()
    for name, param in model.named_modules():
        print(name)