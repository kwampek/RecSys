import numpy as np
from scipy.linalg import solve

from .base import Base


class ImplicitALSRecommender(Base):
    def __init__(self, factors=16, regularization=1., alpha=10., iterations=10,
                 random_state=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state

    def target(self, df):
        return df.recommend.to_numpy(dtype=np.float32)

    def _solve_side(self, observed, target, other):
        gram = other.T @ other + self.regularization * np.eye(self.factors)
        result = np.zeros((observed.shape[0], self.factors))
        for row in range(observed.shape[0]):
            start, end = observed.indptr[row:row + 2]
            ids = observed.indices[start:end]
            sub = other[ids]

            A = gram + self.alpha * (sub.T @ sub)
            b = sub.T @ ((1 + self.alpha) * target.data[start:end])
            result[row] = solve(A, b, assume_a='pos')
        return result

    def fit(self, train_df, games_df=None):
        self._indices(train_df)
        self.observed = self._matrix(train_df, np.ones(len(train_df)))

        self.P = self._matrix(train_df, self.target(train_df))
        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(0, .1, (len(self.items), self.factors))
        for _ in range(self.iterations):
            X = self._solve_side(self.observed, self.P, Y)
            Y = self._solve_side(self.observed.T.tocsr(), self.P.T.tocsr(), X)
        self.user_factors = X.astype(np.float32)
        self.item_factors = Y.astype(np.float32)
        return self

    def score(self, user_id):
        return self.item_factors @ self.user_factors[self.user_to_idx[user_id]]
