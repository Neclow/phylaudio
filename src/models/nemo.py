from .base import BaseFeatureExtractor

import nemo.collections.asr as nemo_asr
import torch


class NeMoFeatureExtractor(BaseFeatureExtractor):
    """NeMo feature extractor

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    """

    def __init__(self, model_id, cache_dir=None, **kwargs):
        super().__init__(model_id)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/NeMo"

        self.load(cache_dir)

    def load(self, cache_dir=None):
        if self.model_id == "NeMo_ambernet":
            model_name = self.model_id.split("_")[-1]

            if cache_dir is None:
                feature_extractor = (
                    nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                        model_name="langid_ambernet"
                    )
                )

            else:
                feature_extractor = (
                    nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
                        restore_path=f"{cache_dir}/{model_name}.nemo"
                    )
                )

            feature_extractor.freeze()

            emb_dim = feature_extractor.decoder.final.in_features
        else:
            raise ValueError(f"Unknown model ID: {self.model_id}")

        self.feature_extractor = feature_extractor
        self.emb_dim = emb_dim

    def forward(self, x):
        # Input shape: B x T
        _, out = self.feature_extractor(
            input_signal=x,
            input_signal_length=torch.tensor([x.shape[1]], device=x.device),
        )

        # Output shape: B x D
        return out

    def get_hidden_states(self, x):
        input_signal_length = torch.tensor([x.shape[1]], device=x.device)

        x, length = self.feature_extractor.preprocessor(
            input_signal=x,
            length=input_signal_length,
        )

        all_hidden_states = []

        # N-1 mega blocks and 1 "epilog" mega block
        # Fig. 1 in https://arxiv.org/pdf/2210.15781
        for block in self.feature_extractor.encoder.encoder[:-1]:
            [x], length = block(([x], length))
            all_hidden_states.append(x)

        # B x n_layers X n_chunks x emb_dim
        hidden_states = torch.stack(all_hidden_states, dim=1).transpose(2, 3)

        return hidden_states
