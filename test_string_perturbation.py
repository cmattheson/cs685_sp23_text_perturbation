

from src.character_perturbation.text_perturbation import TextPerturbationHandler
import os


def main():

    handler = TextPerturbationHandler(
        f'{os.path.dirname(__file__)}/logs/character_perturbation', perturbation_weight=0.8
    )
    test_str = '''CS685 wants us to build a transformer for our final project.'''
    test_str2 = '''This is what a sentence looks like with the new perturbation counts.'''

    result = handler.perturb_string(test_str2)
    print(result)


if __name__ == '__main__':
    main()



