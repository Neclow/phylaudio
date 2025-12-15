from functools import partial

import librosa
import torch
import whisper

from .._config import NONE_TENSOR
from .base import BaseFeatureExtractor, BaseProcessor


class AudioProcessor(BaseProcessor):
    """
    Base processor class for audio models.

    Parameters
    ----------
    sr : int
        Sample rate

    All classes have ```process``` function to process audio files for data loading
    """

    def __init__(self, sr, max_length, **kwargs):
        self.sr = sr
        self.max_length = max_length

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(sr={self.sr}, max_length={self.max_length})"

    def process(self, file_path):
        """
        Process an audio file

        Parameters
        ----------
        file_path : str
            Path to audio file

        Returns
        -------
        torch.Tensor
            Audio waveform loaded as a tensor
        torch.Tensor
            Attention mask (for Transformer-based models), by default a singleton {-1}.
        """
        x, _ = librosa.load(file_path, sr=self.sr)

        return torch.from_numpy(x), NONE_TENSOR


class LogMelSpectrogramFeatureExtractor(BaseFeatureExtractor):
    """Log-mel spectrogram extraction

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    n_mels : int, optional
        Number of mel frequencies
    """

    def __init__(self, model_id, cache_dir=None, n_mels=128, **kwargs):
        super().__init__(model_id)

        self.n_mels = n_mels

        self.load(cache_dir)

    def load(self, cache_dir=None):
        del cache_dir

        self.feature_extractor = partial(
            whisper.log_mel_spectrogram, n_mels=self.n_mels
        )

        self.emb_dim = self.n_mels

    def forward(self, x):
        out = whisper.log_mel_spectrogram(x, n_mels=self.n_mels)

        out = out.permute(0, 2, 1)

        return out
