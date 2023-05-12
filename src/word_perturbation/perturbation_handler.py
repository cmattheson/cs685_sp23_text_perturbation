
# external libraries
import random
import nltk
from random import choice
from nltk.corpus import wordnet
from typing import List, Tuple


class WordPerturbationHandler:

    def __init__(self, perturbation_chance: float = 0.15, verbose: bool = False):

        assert 0 <= perturbation_chance <= 1, 'Perturbation chance needs to be between 0 and 1'
        self.perturbation_chance = perturbation_chance
        self.verbose = verbose
        self.replaceable_parts_of_speech = [
            'CD', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS'
        ]

        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

    def generate_next_word(self, w: str, part_of_speech: str):

        r = random.random()
        next_word = w
        if r <= self.perturbation_chance and part_of_speech in self.replaceable_parts_of_speech:
            synonyms = self.generate_all_synonyms_for_word(w)
            next_word = choice(synonyms)

        return next_word

    @staticmethod
    def generate_unique_word_pos_pairs(sequence: str):

        unique_word_tokens = nltk.word_tokenize(sequence)
        tagged_pos_pairs: List[Tuple[str, str]] = nltk.pos_tag(unique_word_tokens)

        return tagged_pos_pairs

    @staticmethod
    def generate_all_synonyms_for_word(w: str) -> List[str]:

        synonyms = []
        all_wordnet_synonyms = [
            [lemma.name() for lemma in synset.lemmas()]
            for synset in wordnet.synsets(w)
        ]
        for synonym_group in all_wordnet_synonyms:
            synonyms += synonym_group
        synonyms = list(set(synonyms))
        synonyms = [s for s in synonyms if s != w]

        if len(synonyms) == 0:
            synonyms += w

        return synonyms

    def perturb_text_with_synonyms(self, sequence: str) -> str:

        if self.verbose:
            print(sequence)

        sample_pos_tag = self.generate_unique_word_pos_pairs(sequence=sequence)

        new_words = [
            self.generate_next_word(w, pos) for w, pos in sample_pos_tag
        ]

        return ' '.join(new_words)


if __name__ == '__main__':
    test_s = '''
    friend lent dvd got director festival think went warned technical aspects movie bit shaky 
    writing good great maybe colored judgment admit liked moviethe standouts actors youssef 
    kerkor really good ernie main character kind pathetic likable way adam jones also directed 
    justin lane excellent roommates drive ernie mad bill character justin lane spends lot film 
    dressed like panda far favorite seemed least onedimensional reminded old college roommate 
    much called guy watching dvd really kind lovable funny acting good soso none bad also really 
    liked vigilante duo ridiculous funnyim giving one high marks even though issues tell watch 
    people cared decided make movie way well done adam jones crew'''
    repeated_s = 'movie movie movie movie movie movie movie'

    handler = WordPerturbationHandler(perturbation_chance=0.5)

    print('original sentence:', test_s)
    print('perturbed sentence:', handler.perturb_text_with_synonyms(test_s))
    print('original sentence:', repeated_s)
    print('perturbed sentence:', handler.perturb_text_with_synonyms(repeated_s))
