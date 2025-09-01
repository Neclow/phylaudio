import torch
import torch.nn.functional as F
import whisper

from .._config import NONE_TENSOR
from .audio import AudioProcessor
from .base import BaseFeatureExtractor


class WhisperProcessor(AudioProcessor):
    """
    Processor class for Whisper models.

    Parameters
    ----------
    sr : int
        Sample rate
    """

    def process(self, file_path):
        x = whisper.load_audio(file_path, sr=self.sr)

        return whisper.pad_or_trim(torch.from_numpy(x)), NONE_TENSOR


class WhisperFeatureExtractor(BaseFeatureExtractor):
    """Whisper feature extractor

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    """

    def __init__(
        self, model_id, cache_dir=None, training=False, device="cpu", **kwargs
    ):
        dtype = torch.float32 if training or "cuda" not in device else torch.float16

        super().__init__(model_id, dtype)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/whisper"

        self.load(cache_dir)

        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.feature_extractor.is_multilingual,
            num_languages=self.feature_extractor.num_languages,
        )

    def load(self, cache_dir=None):
        model_size = self.model_id.split("-")[-1]

        if "turbo" in model_size:
            self.n_mels = 128
        else:
            self.n_mels = 80

        self.feature_extractor = whisper.load_model(model_size, download_root=cache_dir)

        self.emb_dim = self.feature_extractor.decoder.ln.weight.shape[0]

    def lms(self, x):
        """Log-mel-spectrogram

        Parameters
        ----------
        x : torch.Tensor
            Input of size (B, T)
            B: batch size
            T: sample length

        Returns
        -------
        x : torch.Tensor
            Output of size (B, n_mels, n_frames)
        """
        # Pad so that input is valid for whisper models
        x = whisper.pad_or_trim(x)

        # B x T --> B x M x N
        x = whisper.log_mel_spectrogram(x, n_mels=self.n_mels)

        return x

    def encode(self, x, output_hidden_states=False):
        """
        Forward method of whisper.AudioEncoder with addition of hidden_states

        Adapted from
        https://github.com/openai/whisper/blob/cdb81479623391f0651f4f9175ad986e85777f31/whisper/model.py#L188
        """
        encoder = self.feature_extractor.encoder

        x = F.gelu(encoder.conv1(x))
        x = F.gelu(encoder.conv2(x))
        x = x.permute(0, 2, 1)

        assert (
            x.shape[1:] == encoder.positional_embedding.shape
        ), "incorrect audio shape"
        x = (x + encoder.positional_embedding).to(x.dtype)

        if output_hidden_states:
            hidden_state_list = []

        for block in encoder.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_state_list.append(x)

        x = encoder.ln_post(x)

        if output_hidden_states:
            # B x n_layers x n_chunks x emb_dim
            hidden_states = torch.stack(hidden_state_list, dim=1)

            return x, hidden_states

        return x, None

    def decode(self, x, kv_cache=None, output_hidden_states=False):
        """Forward method of whisper.TextDecoder with addition of hidden_states

        Adapted from
        https://github.com/openai/whisper/blob/cdb81479623391f0651f4f9175ad986e85777f31/whisper/model.py#L227
        """
        n_audio = x.shape[0]

        xt = torch.tensor([[self.tokenizer.sot]] * n_audio).to(x.device)

        decoder = self.feature_extractor.decoder

        # From whisper.TextDecoder.forward
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        xt = (
            decoder.token_embedding(xt)
            + decoder.positional_embedding[offset : offset + xt.shape[-1]]
        )
        xt = xt.to(x.dtype)

        if output_hidden_states:
            hidden_state_list = [xt]

        for block in decoder.blocks:
            xt = block(xt, x, mask=decoder.mask, kv_cache=kv_cache)
            if output_hidden_states:
                hidden_state_list.append(xt)

        xt = decoder.ln(xt)

        if output_hidden_states:
            # B x n_layers x 1 x emb_dim
            hidden_states = torch.stack(hidden_state_list, dim=1)

            return xt, hidden_states

        return xt, None

    def forward(self, x, kv_cache=None):
        # Output shape: B x D
        x = self.lms(x.to(self.dtype))

        x, _ = self.encode(x)

        out, _ = self.decode(x, kv_cache=kv_cache)

        return out[:, 0, :]

    def get_hidden_states(self, x, kv_cache=None):
        x = self.lms(x.to(self.dtype))

        x, encoder_hidden_states = self.encode(x, output_hidden_states=True)

        _, decoder_hidden_states = self.decode(
            x, kv_cache=kv_cache, output_hidden_states=True
        )

        # encoder: B x n_layers x n_chunks x emb_dim
        hidden_states = torch.cat(
            [
                encoder_hidden_states,
                decoder_hidden_states.expand(
                    -1, -1, encoder_hidden_states.shape[2], -1
                ),
            ],
            dim=1,
        )

        return hidden_states


WHISPER_MODELS = {
    k: {"extractor": WhisperFeatureExtractor, "processor": WhisperProcessor}
    for k in (
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3-turbo",
    )
}
