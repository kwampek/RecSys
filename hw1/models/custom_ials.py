import numpy as np

from .implicit_als import ImplicitALSRecommender


class CustomTargetIALS(ImplicitALSRecommender):
    """Recency target, with confidence exactly equal to baseline iALS."""
    def __init__(self, half_life_days=90., **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def target(self, df):
        age_days = (df.post_date.max() - df.post_date).dt.total_seconds() / 86400
        return (df.recommend.to_numpy() * np.exp2(-age_days / self.half_life_days)).astype(np.float32)
