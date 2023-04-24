
# external libraries
import pandas as pd
import os
import yaml

"""
how to interpret each type of perturbation
 - check if it is the exact same character at the location in the "golden" text. If so, all good
 - insert: is the current char not correct and the next char is correct or not
 also confirm that it's not a transposition; maybe do transpose first, then replacement, then delete
 - delete: is the next character not the correct character and the length of the word has changed
 - transpose: is the next character the correct character and the current char the next correct char?
 - replacement: the next character not the correct character and the length of the word has not changed
"""


class PerturbationCalculator:

    def __init__(self, log_directory: str, default_cnt: int = 0, alphanumerics_only: bool = False):

        self.total_chars = 0
        self.log_directory = log_directory

        self.insert_count = 0
        self.insert_matrix = CharMatrixHandler(
            matrix_fp=f'{log_directory}/insert_matrix.csv',
            default_cnt=default_cnt,
            alphanumerics_only=alphanumerics_only
        )

        self.delete_count = 0
        self.delete_matrix = CharMatrixHandler(
            matrix_fp=f'{log_directory}/delete_matrix.csv',
            default_cnt=default_cnt,
            alphanumerics_only=alphanumerics_only
        )

        self.transpose_count = 0
        self.transpose_matrix = CharMatrixHandler(
            matrix_fp=f'{log_directory}/transpose_matrix.csv',
            default_cnt=default_cnt,
            alphanumerics_only=alphanumerics_only
        )

        self.replace_count = 0
        self.replace_matrix = CharMatrixHandler(
            matrix_fp=f'{log_directory}/replace_matrix.csv',
            default_cnt=default_cnt,
            alphanumerics_only=alphanumerics_only
        )

    def ingest_perturbed_text_pairs(self, fp: str):

        text_pair_df = pd.read_csv(fp, sep=',', header=0)
        self.run_all_text_pairs(text_pair_df)
        self.store_results()

    def run_all_text_pairs(self, text_pair_df: pd.DataFrame):

        for i in range(len(text_pair_df)):
            golden_text = text_pair_df.iloc[i, 0]
            perturbed_text = text_pair_df.iloc[i, 1]
            self.check_for_text_pair_perturbations(golden_text, perturbed_text)

    def check_for_text_pair_perturbations(self, golden_text: str, perturbed_text: str):

        golden_idx = 0
        perturbed_idx = 0
        max_golden_idx = len(golden_text)
        max_perturbed_idx = len(perturbed_text)

        while golden_idx < max_golden_idx and perturbed_idx < max_perturbed_idx:
            self.total_chars += 1

            is_correct_char = golden_text[golden_idx] == perturbed_text[perturbed_idx]
            if is_correct_char:
                golden_idx += 1
                perturbed_idx += 1
                continue

            is_transposed_char = (
                golden_text[golden_idx] == perturbed_text[perturbed_idx+1]
                and golden_text[golden_idx+1] == perturbed_text[perturbed_idx]
            )
            if is_transposed_char:
                self.transpose_count += 1
                self.transpose_matrix.add_one(golden_text[golden_idx], golden_text[golden_idx+1])
                golden_idx += 2
                perturbed_idx += 2
                continue

            is_replaced_char = (
                golden_text[golden_idx] != perturbed_text[perturbed_idx]
                and golden_text[golden_idx+1] == perturbed_text[perturbed_idx+1]
            )
            if is_replaced_char:
                self.replace_count += 1
                self.replace_matrix.add_one(golden_text[golden_idx], perturbed_text[perturbed_idx])
                golden_idx += 1
                perturbed_idx += 1
                continue

            is_deleted_char = (
                golden_text[golden_idx] != perturbed_text[perturbed_idx]
                and golden_text[golden_idx+1] == perturbed_text[perturbed_idx]
            )
            if is_deleted_char:
                self.delete_count += 1
                self.delete_matrix.add_one(perturbed_text[perturbed_idx], perturbed_text[perturbed_idx])
                golden_idx += 1
                continue

            is_inserted_char = (
                golden_text[golden_idx] != perturbed_text[perturbed_idx]
            )
            if is_inserted_char:
                self.insert_count += 1
                self.insert_matrix.add_one(perturbed_text[perturbed_idx+1], perturbed_text[perturbed_idx])
                perturbed_idx += 1
                continue

    def write_counts_to_yaml(self):

        counts_dct = {
            'total_chars': self.total_chars,
            'insert_count': self.insert_count,
            'delete_count': self.delete_count,
            'replace_count': self.replace_count,
            'transpose_count': self.transpose_count
        }
        with open(f'{self.log_directory}/char_counts.yml', mode='w') as file:
            yaml.dump(counts_dct, stream=file)

    def store_results(self):
        # TODO: write out each table and a yaml file of the counts of different perturbations
        self.insert_matrix.write_matrix()
        self.delete_matrix.write_matrix()
        self.replace_matrix.write_matrix()
        self.transpose_matrix.write_matrix()
        self.write_counts_to_yaml()


class CharMatrixHandler:

    """
    contains all functions necessary across all the different handlers
    """

    def __init__(self, matrix_fp: str, default_cnt: int, alphanumerics_only: bool):

        self.matrix_fp = matrix_fp
        self.default_cnt = default_cnt
        self.alphanumerics_only = alphanumerics_only

        self.matrix = self.generate_base_matrix(default_cnt=default_cnt, alphanumerics_only=alphanumerics_only)

    @staticmethod
    def generate_base_matrix(default_cnt: int = 0, alphanumerics_only: bool = True) -> pd.DataFrame:

        ascii_chars = [chr(i) for i in range(33, 127)]
        alphanumeric_chars = [
            *[chr(i) for i in range(48, 58)],
            *[chr(i) for i in range(65, 91)],
            *[chr(i) for i in range(97, 123)]
        ]

        if alphanumerics_only:
            labels = alphanumeric_chars
        else:
            labels = ascii_chars

        base_data = [
            [default_cnt for _ in range(len(labels))]
            for _ in range(len(labels))
        ]
        df = pd.DataFrame(base_data, index=labels, columns=labels)

        return df

    def write_matrix(self):

        self.matrix.to_csv(self.matrix_fp, sep=',', index=True, header=True, mode='w')

    def add_one(self, expected_char: str, perturbed_char: str):

        try:
            self.matrix.loc[expected_char, perturbed_char] += 1
        except KeyError:
            pass
