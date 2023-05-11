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

from random import randint, seed, choice
from nltk.corpus import wordnet


# seed(1)


# --------------------------------------------------------------------------------------------------------------------


class SynonymReplacementHandler:

    def __init__(self, perturbation_chance: float = 0.15, verbose=False):
        self.synonyms: dict[str, set[str]] = {}
        self.gathered_synonyms: set = set()
        self.perturbation_chance = perturbation_chance
        self.verbose = verbose

    # This is the main function wich needs to be called. Will take string as input and gives pertub string as output.
    # s = string and tries to perturb each word in the string with a probability of perturbation_chance
    def sentence_perturbe(self, s: str):
        """

        Args:
            s:
            perturbation_chance: chance to perturb a character
            verbose: whether to print logging information

        Returns:

        """

        if (self.verbose):
            print('original sentence:', s)

        # sample_label = row[1]
        sample_tokenized = nltk.word_tokenize(s)
        sample_pos_tag = nltk.pos_tag(sample_tokenized)

        # list of original words that can be replaced
        can_be_replaced_list = []

        new_words = []
        for i in range(0, len(sample_pos_tag)):
            current_word = sample_pos_tag[i][0]
            # check if we have already gathered synonyms for this word
            if current_word not in self.gathered_synonyms:
                if (sample_pos_tag[i][1] in ('CD', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR',
                                     'RBS')):  # ----- Replace the word if it is a noun, adjective, or adverb
                    for syn in wordnet.synsets(current_word):
                        for l in syn.lemmas():
                            if self.verbose:
                                print('synonym:', l.name())
                            # check that the synonym is not the same as the word
                            if current_word != l.name():
                                # check if the key for this word exists in the dictionary
                                if current_word not in self.synonyms:
                                    self.synonyms[current_word] = set()  # create a set of synonyms for this word
                                self.synonyms[current_word].add(l.name())
                                if sample_pos_tag[i][0] not in can_be_replaced_list:
                                    can_be_replaced_list.append(sample_pos_tag[i][0])
            self.gathered_synonyms.add(current_word)  # ----- Mark the word as already gathered
        for word in sample_tokenized:
            r = random.random()
            if r <= self.perturbation_chance and word in self.synonyms.keys():
                new_words.append(choice([*self.synonyms[word]]))
            else:
                new_words.append(word)

        if self.verbose:
            print('perturbed sentence:', ' '.join(new_words))

        return ' '.join(new_words)


if __name__ == '__main__':
    handler = SynonymReplacementHandler(perturbation_chance=1)
    s = "friend lent dvd got director festival think went warned technical aspects movie bit shaky writing good great " \
        "maybe colored judgment admit liked moviethe standouts actors youssef kerkor really good ernie main character " \
        "kind pathetic likable way adam jones also directed justin lane excellent roommates drive ernie mad bill " \
        "character justin lane spends lot film dressed like panda far favorite seemed least onedimensional reminded " \
        "old college roommate much called guy watching dvd really kind lovable funny acting good soso none bad also " \
        "really liked vigilante duo ridiculous funnyim giving one high marks even though issues tell watch people " \
        "cared decided make movie way well done adam jones crew"
    print('original sentence:', s)
    print('perturbed sentence:', handler.sentence_perturbe(s))
    s2 = "movie movie movie movie movie movie movie"
    print('original sentence:', s)
    print('perturbed sentence:', handler.sentence_perturbe(s2))