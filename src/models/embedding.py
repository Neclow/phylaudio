import torch

from ._base import BaseFeatureExtractor


class EmbeddingFeatureExtractor(BaseFeatureExtractor):
    """Pass-through "extractor" over precomputed (cached) embeddings.

    ``forward(x)`` returns ``x`` unchanged, so the LID loop (or any consumer)
    can run on cached embeddings with no other changes -- only the feature
    extractor differs (cached embeddings vs a real backbone).

    Parameters
    ----------
    emb_dim : int
        Embedding dimension (so downstream heads size themselves correctly).
    model_id : str, optional
        Identifier for logging, by default "embedding".
    dtype : torch.dtype, optional
        Embedding dtype, by default torch.float32.
    """

    def __init__(self, emb_dim, model_id="embedding", dtype=torch.float32):
        super().__init__(model_id=model_id, dtype=dtype)

        self.emb_dim = emb_dim

    def load(self, cache_dir=None):
        # Nothing to load: embeddings are precomputed.
        pass

    def forward(self, x, a=None):
        return x
