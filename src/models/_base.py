from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseProcessor(ABC):
    """
    Base processor class for models.

    All classes have ```process``` function to process inputs for data loading
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, x):
        raise NotImplementedError


class BaseFeatureExtractor(nn.Module, ABC):
    """Base audio/text feature extractor

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    """

    def __init__(self, model_id, dtype=torch.float32, **kwargs):
        super().__init__()

        self.model_id = model_id

        self.dtype = dtype

    @abstractmethod
    def load(self, cache_dir=None):
        """Load a pre-trained model

        Parameters
        ----------
        cache_dir : str, optional
            Path to the folder where cached files are stored, by default None
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input of size (B, T)
            B: batch size
            T: sample length
        """
        raise NotImplementedError

    def get_hidden_states(self, x):
        """Get intermediate outputs (hidden states) of a model

        Parameters
        ----------
        x : torch.Tensor
            Input of size (B, T)
            B: batch size
            T: sample length
        """
        raise NotImplementedError
