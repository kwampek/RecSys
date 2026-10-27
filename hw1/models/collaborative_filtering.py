import numpy as np
from scipy import sparse

from .base import Base


class ItemKNNRecommender(Base):
    def __init__(self, n_neighbors=40, block_size=128):
        self.n_neighbors = n_neighbors
        self.block_size = block_size

    def fit(self, train_df, games_df=None):
        self._indices(train_df)
        self.R = self._positive_matrix(train_df)
        norms = np.sqrt(self.R.power(2).sum(axis=0)).A1
        X = self.R @ sparse.diags(1 / np.maximum(norms, 1))
        rows, cols, values = [], [], []
        # Keep only top neighbours; never materialise the entire dense Gram matrix.
        for start in range(0, len(self.items), self.block_size):
            block = (X[:, start:start + self.block_size].T @ X).tocsr()
            for j in range(block.shape[0]):
                row = block.getrow(j)
                ids, sims = row.indices, row.data
                keep = (ids != start + j) & (sims > 0)
                ids, sims = ids[keep], sims[keep]
                top = np.lexsort((ids, -sims))[:self.n_neighbors]
                rows.extend([start + j] * len(top))
                cols.extend(ids[top])
                values.extend(sims[top])
        self.similarity = sparse.csr_matrix((values, (rows, cols)),
                                            shape=(len(self.items), len(self.items)), dtype=np.float32)
        return self

    def score(self, user_id):
        return (self.R.getrow(self.user_to_idx[user_id]) @ self.similarity).toarray().ravel()
