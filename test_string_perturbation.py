

from src.character_perturbation.text_perturbation import TextPerturbationHandler
import os


def main():

    handler = TextPerturbationHandler(f'{os.path.dirname(__file__)}/logs')
    test_str = '''CS685 wants us to build a transformer for our final project.'''

    result = handler.perturb_string(test_str)
    print(result)


if __name__ == '__main__':
    main()



