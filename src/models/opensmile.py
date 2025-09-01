import opensmile
import torch

from .._config import SAMPLE_RATE
from .base import BaseFeatureExtractor


class openSmileFeatureExtractor(BaseFeatureExtractor):
    """openSMILE-based feature extractor

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    cache_dir : str, optional
        Unused
    """

    def __init__(self, model_id, cache_dir=None, **kwargs):
        super().__init__(model_id)

        self.load(cache_dir)

    def load(self, cache_dir=None):
        feature_set = opensmile.FeatureSet[self.model_id.split("openSMILE/")[-1]]

        self.feature_extractor = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self.emb_dim = self.feature_extractor.num_features

    def forward(self, x):
        out = torch.zeros(len(x), self.emb_dim)
        for i, x_i in enumerate(x):
            out[i] = torch.from_numpy(
                self.feature_extractor.process_signal(
                    x_i, sampling_rate=SAMPLE_RATE
                ).values
            )
        return out
