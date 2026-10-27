"""Shared indexing and deterministic ranking for already cleaned interactions."""
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse


class Base(ABC):
    def _indices(self, df):
        self.users = sorted(df.user.unique())
        self.items = sorted(df.AppID.unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {it: i for i, it in enumerate(self.items)}
        self.seen = df.groupby('user').AppID.agg(set).to_dict()

    def _matrix(self, df, values):
        return sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32),
             (df.user.map(self.user_to_idx), df.AppID.map(self.item_to_idx))),
            shape=(len(self.users), len(self.items)),
        )

    def _positive_matrix(self, df):
        positive = df[df.recommend]
        return self._matrix(positive, np.ones(len(positive)))

    @abstractmethod
    def fit(self, train_df, games_df=None): ...

    @abstractmethod
    def score(self, user_id): ...

    def recommend(self, user_id, k=10, candidates=None):
        if user_id not in self.user_to_idx or k <= 0:
            return []
        allowed = set(self.items if candidates is None else candidates)
        indices = np.array([i for i, it in enumerate(self.items)
                            if it in allowed and it not in self.seen[user_id]], dtype=int)
        scores = self.score(user_id)

        top = indices[np.argsort(-scores[indices], kind='stable')[:k]]
        return [(self.items[i], float(scores[i])) for i in top]
