import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data.datasets import PerturbedSequenceDataset
from src.models.bert_models import *
import torch.nn as nn
from data.datasets import *
from transformers import BertModel
import editdistance
import random
import pandas as pd
from collections import OrderedDict

from src.training.training import compute_statistics






def run_final_ag_news_experiments(names, models, word_perturbation_rates, char_perturbation_rates,
                                  device='cuda') -> None:
    # evaluate on test set with char perturbation and word perturbation
    for char_perturbation_rate in char_perturbation_rates:
        for word_perturbation_rate in word_perturbation_rates:
            test_set = PerturbedSequenceDataset(test_data['text'], torch.Tensor(test_data['label']).to(torch.long),
                                                val_char_perturbation_rate=char_perturbation_rate,
                                                val_word_perturbation_rate=word_perturbation_rate)
            test_set.eval()  # very important: set this to eval so we use the val perturbation rates
            dataloader = DataLoader(test_set, batch_size=32, num_workers=2)
            for name, model in zip(model_names, models):
                model.to(device)
                loss, accuracy = compute_statistics(model, criterion, dataloader)
                print(f'Model name: {name}')
                print(f'char perturbation rate: {char_perturbation_rate}, word perturbation rate: {word_perturbation_rate}')
                print(f'test loss: {loss}, test accuracy: {accuracy}')

def load_state_fix_params(model, path):
    """
    This function is a hack to correctly load the state dict of models that were saved with an older version of the 
    code so that we can evaluate them on test. I hate it as much as you do. 
    
    Args: model: the model to load the state dict into
    path: the path to the state dict to load

    Returns:

    """
    try:
        model.load_state_dict(torch.load(
            path))
    except:
        # fix the state dict
        state_dict = torch.load('src/models/model.pt')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'encoder.elmo_embedding' in k:
                new_k = k.replace('encoder.elmo_embedding', 'encoder.elmo_embedder')
                new_state_dict[new_k] = v
            elif 'encoder.linear' in k:
                # the linear parameters don't do anything, ignore them; they're deleted in the latest model version.
                continue
            else:
                new_state_dict[k] = v
        if 'encoder.elmo_layernorm.weight' not in new_state_dict.keys():
            # add the useless layernorm parameters. Yes, it's a hack, but it makes the model load correctly.
            new_state_dict['encoder.elmo_layernorm.weight'] = torch.ones(768)
            new_state_dict['encoder.elmo_layernorm.bias'] = torch.zeros(768)
            model.encoder.layernorm_elmo_separately = False  # make sure the model doesn't try to actually use
            # these parameters since the model the state dict was saved from doesn't use them
        model.load_state_dict(new_state_dict)

if __name__ == '__main__':
    """
    all_data = load_dataset('src/data/ag_news.py')
    # test_data = random.choices(all_data['test']['text'], k=100)
    test_data = all_data['test']['text']
    cleaned_test_idxs = []
    num_similar = 0

    for i, test_str in enumerate(test_data):
        found_similar = False
        if i % 1 == 0:
            print(f'test string {i + 1} / {len(test_data)}')
        for train_str in all_data['train']['text']:
            if editdistance.eval(test_str, train_str) < 100:
                num_similar += 1
                found_similar = True
                break
        if not found_similar:
            cleaned_test_idxs.append(i)

        # put the cleaned test data into a dataframe
    df = pd.DataFrame({'text': [test_data[i] for i in cleaned_test_idxs],
                       'label': [all_data['test']['label'][i] for i in cleaned_test_idxs]})
    # show the dataframe
    print(df)
    # show the length of the dataframe
    print(len(df))
    # save the dataframe to disk
    df.to_csv('src/data/datasets/ag_news_cleaned_test.csv', index=False)

    print(f'num similar: {num_similar}')
    
    ok I cleaned the data so now this isn't needed but it can stay here I guess
    """

    test_data = pd.read_csv('src/data/datasets/ag_news_cleaned_test.csv')
    state_dict = torch.load('src/models/model.pt')
    model2 = ClassifierModel(Bert_Plus_Elmo_Concat(), classifier=nn.Linear(768, 4))

    load_state_fix_params(model, 'src/models/model.pt')

    #model2 = ClassifierModel(Bert_Plus_Elmo_Concat(layer_norm_elmo_separately=True), classifier=nn.Linear(768, 4))
    #model2.load_state_dict(torch.load(
    #    'src/models/pretrained/ag_news_concatenated_bert_elmo_model_char_5.0_perturbed_hyperparameter_optimization_lr_3e-05/model.pt'))
    model_names = 'concat_model', 'additive_model'

    models = [model]
    char_perturbation_rates = [0, 1, 2, 3, 4, 5]
    criterion = torch.nn.functional.cross_entropy
    device = 'cuda'
    for perturbation_rate in char_perturbation_rates:
        test_set = PerturbedSequenceDataset(test_data['text'], torch.Tensor(test_data['label']).to(torch.long),
                                            val_char_perturbation_rate=perturbation_rate)
        test_set.eval()  # very important: set this to eval so we use the val perturbation rates
        dataloader = DataLoader(test_set, batch_size=32, num_workers=2)
        for name, model in zip(model_names, models):
            model.to(device)
            loss, accuracy = compute_statistics(model, criterion, dataloader)
            print(loss, accuracy)
