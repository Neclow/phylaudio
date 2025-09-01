import torch
import torchaudio
from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.inference.classifiers import EncoderClassifier

from .._config import NONE_TENSOR
from .audio import AudioProcessor
from .base import BaseFeatureExtractor


class SpeechbrainProcessor(AudioProcessor):
    """
    Processor class for Speechbrain models.

    Parameters
    ----------
    sr : int
        Sample rate
    """

    def __init__(self, sr, **kwargs):
        super().__init__(sr)

        self.audio_normalizer = AudioNormalizer(sample_rate=self.sr)

    def process(self, file_path):
        x, sr = torchaudio.load(file_path, channels_first=False)

        x = self.audio_normalizer(x, sr)

        return x, NONE_TENSOR


class SpeechbrainFeatureExtractor(BaseFeatureExtractor):
    """Speechbrain feature extractor

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    device : str, optional
        Device on which torch tensors will be loaded, by default 'cpu'
    """

    def __init__(self, model_id, cache_dir=None, device="cpu", **kwargs):
        super().__init__(model_id)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/huggingface"

        self.load(cache_dir, device)

        self.load_audio = self.feature_extractor.load_audio

    def load(self, cache_dir=None, device="cpu"):
        if self.model_id == "speechbrain/lang-id-voxlingua107-ecapa":
            self.feature_extractor = EncoderClassifier.from_hparams(
                source=self.model_id, savedir=cache_dir, run_opts={"device": device}
            )
            self.emb_dim = 256
        else:
            raise ValueError(f"Unknown model ID: {self.model_id}")

    def forward(self, x):
        # Input shape: B x T
        # Output shape: B x D
        return self.feature_extractor.encode_batch(x)[:, 0, :]

    def get_hidden_states(self, x):
        """Get all hidden states from speechbrain models

        For ECAPA-TDNN: the function is equivalent to the forward method of
        speechbrain.lobes.models.ECAPA_TDNN with addition of hidden_states

        Copyright (c) 2020 Speechbrain
        """
        mods = self.feature_extractor.mods

        wav_lens = torch.ones(x.shape[0], device=x.device)

        x = mods.compute_features(x)

        x = mods.mean_var_norm(x, wav_lens)

        x = x.transpose(1, 2)

        xl = []

        for block in mods.embedding_model.blocks:
            x = block(x)
            xl.append(x)

        hidden_states = torch.stack(xl[1:], dim=1).transpose(2, 3)

        return hidden_states
