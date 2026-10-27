import numpy as np
from scipy.linalg import solve

from .base import Base


class EASERecommender(Base):
    def __init__(self, regularization=50., max_items=1000):
        self.regularization = regularization
        self.max_items = max_items

    def fit(self, train_df, games_df=None):
        self._indices(train_df)
        self.R = self._positive_matrix(train_df)
        gram = (self.R.T @ self.R).toarray().astype(np.float64)
        gram.flat[::len(self.items) + 1] += self.regularization
        precision = solve(gram, np.eye(len(self.items)), assume_a='pos')
        self.B = (-precision / np.diag(precision)[None, :]).astype(np.float32)
        np.fill_diagonal(self.B, 0)
        return self

    def score(self, user_id):
        return (self.R.getrow(self.user_to_idx[user_id]) @ self.B).ravel()
