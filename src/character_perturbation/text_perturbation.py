

import pandas as pd
import random
import yaml


class CharacterInsertionGenerator:

    def __init__(self, log_directory: str):

        self.log_directory = log_directory
        self.suffix = 'insert_matrix'
        self.matrix = pd.read_csv(f'{log_directory}/{self.suffix}.csv', sep=',', header=0, index_col=0)
        self.convert_matrix_to_percentages()

    def convert_matrix_to_percentages(self):

        row_counts = [sum(self.matrix.iloc[i, :]) for i in range(len(self.matrix))]
        self.matrix = self.matrix.div(row_counts, axis='rows')

    def get(self, c: str) -> str:

        """
        input: character output: character
        take a character and filter down to the row of the character you want; convert to dictionary
        convert counts to percentages at initialization
        run the same random process I did at the parent level across what character should be returned
        have a small percent be a completely random character
        """

        row = self.matrix.loc[c, :]
        r = random.random()
        idx = -1
        while r > 0:
            idx += 1
            r -= row[idx]

        return self.matrix.columns[idx]


class CharacterReplacementGenerator:

    def __init__(self, log_directory: str):

        self.log_directory = log_directory
        self.suffix = 'replace_matrix'
        self.matrix = pd.read_csv(f'{log_directory}/{self.suffix}.csv', sep=',', header=0, index_col=0)
        self.convert_matrix_to_percentages()

    def convert_matrix_to_percentages(self):

        row_counts = [sum(self.matrix.iloc[i, :]) for i in range(len(self.matrix))]
        self.matrix = self.matrix.div(row_counts, axis='rows')

    def get(self, c: str) -> str:

        """
        input: character output: character
        take a character and filter down to the row of the character you want; convert to dictionary
        convert counts to percentages at initialization
        run the same random process I did at the parent level across what character should be returned
        have a small percent be a completely random character
        """

        row = self.matrix.loc[c, :]
        r = random.random()
        idx = -1
        while r > 0:
            idx += 1
            r -= row[idx]

        return self.matrix.columns[idx]


class TextPerturbationHandler:

    def __init__(self, log_directory: str):

        perturb_cnts = self.load_logged_perturbation_counts(log_directory=log_directory)
        self.total_chars = int(perturb_cnts.get('total_chars'))

        self.insert_cnt = int(perturb_cnts.get('insert_count'))
        self.pct_insert = self.insert_cnt / self.total_chars

        self.delete_cnt = int(perturb_cnts.get('delete_count'))
        self.pct_delete = self.delete_cnt / self.total_chars

        self.transpose_cnt = int(perturb_cnts.get('transpose_count'))
        self.pct_transpose = self.transpose_cnt / self.total_chars

        self.replace_cnt = int(perturb_cnts.get('replace_count'))
        self.pct_replace = self.replace_cnt / self.total_chars

        self.insert_generator = CharacterInsertionGenerator(log_directory=log_directory)
        self.replacement_generator = CharacterReplacementGenerator(log_directory=log_directory)

    def load_logged_perturbation_counts(self, log_directory: str) -> dict:

        with open(f'{log_directory}/char_counts.yml', mode='r') as file:
            perturb_cnts = yaml.load(file, yaml.BaseLoader)

        return perturb_cnts

    def assert_colname(self, df: pd.DataFrame):

        colnames = df.columns

        assert 'text' in colnames, f'\'text\' column not present in DataFrame. Columns: {colnames}'

    def transform_text(self, df: pd.DataFrame):

        self.assert_colname(df)

        perturbed_lst = [self.perturb_string(s) for s in df['text']]
        df['perturbed_text'] = perturbed_lst

        return df

    def perturb_string(self, s: str):

        lst = []

        for i in range(len(s)):
            r = random.random()

            if r <= self.pct_insert:
                lst.append(s[i])
                inserted_char = self.insert_generator.get(s[i])
                lst.append(inserted_char)
                continue
            r -= self.pct_insert

            if r <= self.pct_delete:
                continue
            r -= self.pct_delete

            if r <= self.pct_transpose:
                lst.insert(i-1, s[i])
                continue
            r -= self.pct_transpose

            if r <= self.pct_replace:
                replacement_char = self.replacement_generator.get(s[i])
                lst.append(replacement_char)
                continue
            else:
                lst.append(s[i])

        perturbed_s = ''.join(lst)
        return perturbed_s





