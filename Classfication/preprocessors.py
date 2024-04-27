from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Create abstract class with ABC with method "name()"
class Preprocessor(ABC):

    @abstractmethod
    def name(self):
        pass


class BiasPreprocessor(BaseEstimator, TransformerMixin, Preprocessor):
    def name(self):
        return 'bias'

    def __init__(self, replace_husband_wife=False, drop_sex=False, drop_gave_birth=False, drop_occupation=False):
        self.replace_husband_wife = replace_husband_wife
        self.drop_sex = drop_sex
        self.drop_gave_birth = drop_gave_birth
        self.drop_occupation = drop_occupation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.replace_husband_wife:
            X['marital status'] = X['marital status'].replace(['Husband', 'Wife'], 'Married')
        if self.drop_sex:
            X = X.drop(columns=['sex'])
        if self.drop_gave_birth:
            X = X.drop(columns=['gave birth this year'])
        if self.drop_occupation:
            X = X.drop(columns=['occupation'])
        return X

class ImputePreprocessor(BaseEstimator, TransformerMixin, Preprocessor):
    def name(self):
        return 'impute'

    def __init__(self, fill_english=None, fill_gave_birth=None):
        self.fill_english = fill_english
        self.fill_gave_birth = fill_gave_birth

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.fill_english is not None:
            X['ability to speak english'] = X['ability to speak english'].fillna(self.fill_english)
        if self.fill_gave_birth is not None:
            X['gave birth this year'] = X['gave birth this year'].fillna(self.fill_gave_birth)
        return X


class NumCatPreprocessor(ColumnTransformer, Preprocessor):

    def name(self):
        return 'preprocessor'

    def __init__(self, numerical_columns, categorical_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.numerical_transformer = StandardScaler()

        super().__init__(
            transformers=[
                ('num', self.numerical_transformer, self.numerical_columns),
                ('cat', self.categorical_transformer, self.categorical_columns)
            ],
            sparse_threshold=1.0
        )
