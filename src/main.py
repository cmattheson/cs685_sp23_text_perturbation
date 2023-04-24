

import pandas as pd
import random

# build a class that will create the perturbations of the text
# receive a pandas dataframe with each line being the text we want to perturb
# read in two additional pandas dataframes during initialization that contains substitute letters
# and additionally inserted letters

# handling perturbations
# we will iterate over every character in the text and potentially perform a perturbation
# possible perturbations:
#  - insertions
#  - deletions
#  - transpositions
#  - incorrect character
# we will generate a random number for at each character and have a different option for what would
# happen depending on the random number
# these ranges are dependent on receiving percentages from the previous data


class TextPerturbationHandler:

    def __init__(self, training_fp: str):

        self.pct_insert = 0
        self.pct_delete = 0
        self.pct_transpose = 0
        self.pct_replace = 0
        self.insert_generator = None
        self.replacement_generator = None

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
                # TODO: insert additional char after real char
                #  pull from pandas list what that additional character should be
                lst.append(s[i])
                inserted_char = self.insert_generator.get(s[i])
                lst.append(inserted_char)

                r -= self.pct_insert

            elif r <= self.pct_delete:
                r -= self.pct_delete

            elif r <= self.pct_transpose:
                lst.insert(i-1, s[i])
                r -= self.pct_transpose

            elif r <= self.pct_replace:
                # TODO: pull from other pandas list what the new character should be
                replacement_char = self.replacement_generator.get(s[i])
                lst.append(replacement_char)
            else:
                lst.append(s[i])

        perturbed_s = ''.join(lst)
        return perturbed_s


class CharacterInsertionGenerator:

    def get(self, c: str) -> str:

        """
        input: character output: character
        take a character and filter down to the row of the character you want; convert to dictionary
        convert counts to percentages at initialization
        run the same random process I did at the parent level across what character should be returned
        have a small percent be a completely random character
        """

        return 'a'


class CharacterReplacementGenerator:

    def get(self, c: str) -> str:

        """
        input: character output: character
        take a character and filter down to the row of the character you want; convert to dictionary
        convert counts to percentages at initialization
        run the same random process I did at the parent level across what character should be returned
        have a small percent be a completely random character
        """

        return 'a'






