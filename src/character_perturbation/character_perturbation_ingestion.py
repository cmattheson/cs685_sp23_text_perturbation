
import os
from src.character_perturbation.perturbation_rate_calculator import PerturbationCalculator


def main():

    project_root = '/'.join(os.path.dirname(__file__).split(sep='/')[:-2])
    calc = PerturbationCalculator(
        log_directory=f'{project_root}/logs/character_perturbation', default_cnt=0
    )
    calc.ingest_perturbed_text_pairs(fp=f'{project_root}/data/character_perturbation/keystrokes_source_data.csv')
    calc.store_results()


if __name__ == '__main__':
    main()
