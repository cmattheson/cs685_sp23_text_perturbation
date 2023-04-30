
import os
from src.character_perturbation.perturbation_calculator import PerturbationCalculator


def main():

    calc = PerturbationCalculator(log_directory=f'{os.path.dirname(__file__)}/logs', default_cnt=0.05)
    calc.check_for_text_pair_perturbations(
        'the quick brown fox jumped over the lazy dog',
        'the quick brown fox jumped over the lazy dog'
    )
    calc.check_for_text_pair_perturbations(
        'the quick brown fox jumped over the lazy dog',
        'the quickkkkk brown foix julmped ov er the lazy doggo'
    )
    calc.check_for_text_pair_perturbations(
        'the quick brown fox jumped over the lazy dog',
        'th quik brwn fox jumped over the lazy do'
    )
    calc.check_for_text_pair_perturbations(
        'the quick brown fox jumped over the lazy dog',
        'tha qu ck briwn fix jumped over the lazy dog'
    )
    calc.check_for_text_pair_perturbations(
        'the quick brown fox jumped over the lazy dog',
        'the qucik brown foxj umped oevr the lazy dog'
    )
    calc.store_results()


if __name__ == '__main__':
    main()
