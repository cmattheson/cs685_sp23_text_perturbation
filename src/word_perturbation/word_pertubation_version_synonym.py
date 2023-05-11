# -*- coding: utf-8 -*-
"""word_pertubation_version1.0.1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mrGu6z5Sl7-Vglpb5fiZJpBedpKxJwAI
"""
import random

import nltk

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from random import randint, seed, sample, choice
from nltk.corpus import wordnet


# seed(1)


# --------------------------------------------------------------------------------------------------------------------

def return_random_number(begin, end):
    return randint(begin, end)


class Synonym:

    def __init__(self, synonym, original):
        self.synonym = synonym
        self.original = original


# This is the main function wich needs to be called. Will take string as input and gives pertub string as output.
# s = string and tries to perturb each word in the string with a probability of perturbation_chance
def sentence_pertube(s, perturbation_chance: float = 0.15, verbose=False):
    synonyms: dict[str, set[str]] = {}

    if (verbose):
        print(s)

    # sample_label = row[1]
    sample_tokenized = nltk.word_tokenize(s)
    sample_pos_tag = nltk.pos_tag(sample_tokenized)

    # list of original words that can be replaced
    can_be_replaced_list = []

    new_words = []
    for i in range(0, len(sample_pos_tag)):
        if (sample_pos_tag[i][1] in ('CD', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR',
                                     'RBS')):  # ----- Replace the word if it is a noun, adjective, or adverb
            current_word = sample_pos_tag[i][0]

            for syn in wordnet.synsets(current_word):
                for l in syn.lemmas():
                    # check that the synonym is not the same as the word
                    if (sample_pos_tag[i][0] != l.name()):
                        # check if the key for this word exists in the dictionary
                        if not current_word in synonyms:
                            synonyms[current_word] = set()  # create a set of synonyms for this word
                        synonyms[current_word].add(l.name())
                        if (sample_pos_tag[i][0] not in can_be_replaced_list):
                            can_be_replaced_list.append(sample_pos_tag[i][0])
    for word in sample_tokenized:
        r = random.random()
        if r <= perturbation_chance and word in synonyms.keys():
            new_words.append(choice([*synonyms[word]]))
        else:
            new_words.append(word)

    return ' '.join(new_words)


if __name__ == '__main__':
    s = "friend lent dvd got director festival think went warned technical aspects movie bit shaky writing good great maybe colored judgment admit liked moviethe standouts actors youssef kerkor really good ernie main character kind pathetic likable way adam jones also directed justin lane excellent roommates drive ernie mad bill character justin lane spends lot film dressed like panda far favorite seemed least onedimensional reminded old college roommate much called guy watching dvd really kind lovable funny acting good soso none bad also really liked vigilante duo ridiculous funnyim giving one high marks even though issues tell watch people cared decided make movie way well done adam jones crew"
    print('original sentence:', s)
    print('perturbed sentence:', sentence_pertube(s, 0.5))
    s2 = "movie movie movie movie movie movie movie"
    print('original sentence:', s)
    print('perturbed sentence:', sentence_pertube(s2, 0.5))
