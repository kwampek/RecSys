import numpy as np
import pandas as pd
from scipy.special import expit

from .base import Base


class MetadataPreprocessor:
    """Build categorical metadata tokens and a shared feature vocabulary."""

    COLUMNS = ("genres", "tags")

    @staticmethod
    def _as_list(value):
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return []

    def fit(self, games_df):
        games = games_df.copy()
        games["AppID"] = games["AppID"].astype(str)
        games = games.drop_duplicates("AppID").set_index("AppID")

        self.tokens_by_item = {}
        vocabulary = set()

        for item_id, row in games.iterrows():
            tokens = set()

            for column in self.COLUMNS:
                for value in self._as_list(row.get(column)):
                    value = str(value).strip().casefold()
                    if value:
                        tokens.add(f"{column[:-1]}={value}")

            self.tokens_by_item[item_id] = sorted(tokens)
            vocabulary.update(tokens)

        self.vocabulary = {
            token: index for index, token in enumerate(sorted(vocabulary))
        }
        return self

    def transform(self, item_ids):
        return [
            self.tokens_by_item.get(str(item_id), [])
            for item_id in item_ids
        ]


class FactorizationMachineRecommender(Base):
    def __init__(
        self,
        factors=12,
        epochs=15,
        learning_rate=0.02,
        batch_size=256,
        negatives_per_positive=1,
        l2=1e-4,
        random_state=42,
        use_metadata=True,
    ):
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.negatives_per_positive = negatives_per_positive
        self.l2 = l2
        self.random_state = random_state
        self.use_metadata = use_metadata

    def _build_item_features(self, games_df):
        item_tokens = [[] for _ in self.items]

        if self.use_metadata:
            self.preprocessor = MetadataPreprocessor().fit(games_df)
            item_tokens = self.preprocessor.transform(self.items)
            vocabulary = self.preprocessor.vocabulary
        else:
            vocabulary = {}

        n_users = len(self.users)
        n_items = len(self.items)

        metadata_offset = n_users + n_items

        self.item_features = np.zeros(
            (n_items, 1 + max(map(len, item_tokens), default=0)),
            dtype=np.int32,
        )

        for item_idx, tokens in enumerate(item_tokens):
            ids = [n_users + item_idx + 1]
            ids.extend(
                metadata_offset + vocabulary[token] + 1
                for token in tokens
            )
            self.item_features[item_idx, :len(ids)] = ids

        self.n_features = metadata_offset + len(vocabulary) + 1

    def _features(self, users, items):
        return np.column_stack((users, self.item_features[items]))

    def _forward(self, features):
        v = self.v[features]
        summed = v.sum(axis=1)

        linear = self.w[features].sum(axis=1)
        interaction = 0.5 * (
            summed**2 - (v**2).sum(axis=1)
        ).sum(axis=1)

        logits = self.bias[0] + linear + interaction
        return logits, v, summed

    def _sample_training_data(self, train_df, rng):
        users = train_df["user"].map(self.user_to_idx).to_numpy() + 1
        items = train_df["AppID"].map(self.item_to_idx).to_numpy()
        labels = train_df["recommend"].to_numpy(dtype=float)

        sampled_users, sampled_items = [], []

        for user_id, group in train_df.groupby("user", sort=True):
            n_positive = int(group["recommend"].sum())
            n_negative = self.negatives_per_positive * n_positive

            if n_negative <= 0:
                continue

            seen = self.seen[user_id]
            pool = np.array(
                [
                    self.item_to_idx[item]
                    for item in self.items
                    if item not in seen
                ],
                dtype=np.int32,
            )

            n_negative = min(n_negative, len(pool))
            if n_negative == 0:
                continue

            chosen = rng.choice(pool, size=n_negative, replace=False)
            sampled_users.extend([self.user_to_idx[user_id] + 1] * n_negative)
            sampled_items.extend(chosen.tolist())

        if sampled_users:
            users = np.concatenate((users, np.asarray(sampled_users)))
            items = np.concatenate((items, np.asarray(sampled_items)))
            labels = np.concatenate((labels, np.zeros(len(sampled_users))))

        return users.astype(np.int32), items.astype(np.int32), labels

    def fit(self, train_df, games_df=None):
        train_df = train_df.copy()
        train_df["user"] = train_df["user"].astype(str)
        train_df["AppID"] = train_df["AppID"].astype(str)

        if self.use_metadata and games_df is None:
            raise ValueError("games_df is required when use_metadata=True")

        self._indices(train_df)
        self._build_item_features(games_df)

        init_rng = np.random.default_rng(self.random_state)
        sample_rng = np.random.default_rng(self.random_state + 1)
        shuffle_rng = np.random.default_rng(self.random_state + 2)

        self.w = np.zeros(self.n_features)
        self.v = init_rng.normal(
            0, 0.03, size=(self.n_features, self.factors)
        )
        self.bias = np.zeros(1)
        self.w[0] = 0.0
        self.v[0] = 0.0

        users, items, labels = self._sample_training_data(
            train_df, sample_rng
        )

        parameters = [self.w, self.v, self.bias]
        first_moment = [np.zeros_like(p) for p in parameters]
        second_moment = [np.zeros_like(p) for p in parameters]
        step = 0
        self.loss_history = []

        for _ in range(self.epochs):
            order = shuffle_rng.permutation(len(labels))
            epoch_loss = 0.0

            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                batch_labels = labels[batch_idx]
                features = self._features(users[batch_idx], items[batch_idx])

                logits, v, summed = self._forward(features)
                epoch_loss += (
                    np.logaddexp(0, logits).sum()
                    - batch_labels @ logits
                )

                error = (expit(logits) - batch_labels) / len(batch_idx)

                grad_w = np.zeros_like(self.w)
                grad_v = np.zeros_like(self.v)

                np.add.at(
                    grad_w,
                    features.ravel(),
                    np.repeat(error, features.shape[1]),
                )

                contribution = error[:, None, None] * (
                    summed[:, None, :] - v
                )
                np.add.at(
                    grad_v,
                    features.ravel(),
                    contribution.reshape(-1, self.factors),
                )

                grad_w += self.l2 * self.w
                grad_v += self.l2 * self.v
                grad_w[0] = 0.0
                grad_v[0] = 0.0

                gradients = [
                    grad_w,
                    grad_v,
                    np.array([error.sum()]),
                ]

                step += 1
                for param, grad, m, s in zip(
                    parameters, gradients, first_moment, second_moment
                ):
                    m *= 0.9
                    m += 0.1 * grad
                    s *= 0.999
                    s += 0.001 * grad**2

                    m_hat = m / (1 - 0.9**step)
                    s_hat = s / (1 - 0.999**step)

                    param -= self.learning_rate * m_hat / (
                        np.sqrt(s_hat) + 1e-8
                    )

                self.w[0] = 0.0
                self.v[0] = 0.0

            self.loss_history.append(epoch_loss / len(labels))
        return self

    def score(self, user_id):
        user_idx = self.user_to_idx[user_id] + 1
        scores = np.empty(len(self.items))

        for start in range(0, len(self.items), self.batch_size):
            item_idx = np.arange(
                start, min(start + self.batch_size, len(self.items))
            )
            user_ids = np.full(len(item_idx), user_idx)
            features = self._features(user_ids, item_idx)
            scores[item_idx] = self._forward(features)[0]

        return scores
