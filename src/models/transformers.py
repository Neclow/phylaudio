import torch
from transformers import (
    AutoFeatureExtractor,
    T5EncoderModel,
    Wav2Vec2Model,
    XLMRobertaModel,
)

from .audio import AudioProcessor
from .base import BaseFeatureExtractor


class TransformersAudioProcessor(AudioProcessor):
    """
    Processor class for Transformers models.

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    sr : int
        Sample rate
    max_length : int, optional
        Max. number of samples/frames to keep, by default None
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    truncation : bool, optional
        Activates truncation to cut input sequences longer than "max_length" to "max_length"., by default True
    padding : str or bool, optional
        Strategy to pad the returned sequences, by default "max_length"
        "max_length": pad to ```max_length```
        False: no padding
        True or "longest": pad to the longest sequence
    padding_side : str, optional
        Where padding elements are added, by default "right"
    """

    def __init__(
        self,
        sr,
        max_length,
        model_id,
        cache_dir=None,
        truncation=True,
        padding="max_length",
        padding_side="right",
        **kwargs,
    ):
        super().__init__(sr, max_length)

        self.model_id = model_id

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/huggingface"

        self.processor = AutoFeatureExtractor.from_pretrained(
            self.model_id, cache_dir=cache_dir, padding_side=padding_side, **kwargs
        )

        self.truncation = truncation
        self.padding = padding

    def process(self, file_path):
        x, _ = super().process(file_path)

        # # This ensures that the same data is fed in the model
        # # Needed because max_length is not the same for whisper
        x = x[: self.max_length]

        xp = self.processor(
            x,
            sampling_rate=self.sr,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_attention_mask=True,
        )

        if "w2v-bert" in self.model_id:  # or "whisper" in self.model_id:
            x = xp.input_features[0]
        else:
            x = xp.input_values[0]

        a = xp.attention_mask[0]

        return x, a


class TransformersFeatureExtractor(BaseFeatureExtractor):
    """Transformers feature extractor

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    finetuned : bool, optional
        Whether to load a LID-finetuned wav2vec2 or not, by default False
    dtype : torch.dtype, optional
        Torch tensor data type, by default torch.float16
    """

    def __init__(
        self,
        model_id,
        cache_dir=None,
        finetuned=False,
        average_pool=False,
        training=False,
        device="cpu",
        **kwargs,
    ):
        dtype = torch.float32 if training or "cuda" not in device else torch.float16

        super().__init__(model_id, dtype)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/huggingface"

        self.finetuned = finetuned

        self.average_pool = average_pool

        self.load(cache_dir)

    def load(self, cache_dir=None):
        # TODO: use AutoModel?
        if "wav2vec2" in self.model_id or "mms" in self.model_id:
            feature_extractor = Wav2Vec2Model.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                torch_dtype=self.dtype,
                attn_implementation=(
                    "flash_attention_2" if self.dtype == torch.float16 else "eager"
                ),
            )

            if "mms" in self.model_id:
                emb_dim = 1280
            else:
                if self.finetuned:
                    feature_extractor = _load_finetuned_xlsr(feature_extractor)
                emb_dim = 1024

            feature_extractor.freeze_feature_encoder()
        elif "aya" in self.model_id:
            feature_extractor = T5EncoderModel.from_pretrained(
                self.model_id, cache_dir=cache_dir, torch_dtype=self.dtype
            )
            emb_dim = 4096
        elif "xlm" in self.model_id:
            feature_extractor = XLMRobertaModel.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                torch_dtype=self.dtype,
                add_pooling_layer=False,
            )
            emb_dim = 768
        else:
            raise ValueError(f"Unknown model ID: {self.model_id}")

        self.feature_extractor = feature_extractor
        self.emb_dim = emb_dim

    def forward(self, x, a=None):
        # last_hidden_state shape:
        # Audio = B x n_chunks x emb_dim
        # Text = B x n_tokens x emb_dim
        if x.dtype != torch.long:
            x = x.to(self.dtype)

        last_hidden_state = self.feature_extractor(
            x, attention_mask=a
        ).last_hidden_state

        if self.average_pool:
            out = last_hidden_state.mean(dim=1)
        else:
            if "wav2vec2" in self.model_id or "mms" in self.model_id:
                # Source:
                # https://github.com/huggingface/transformers/blob/ccbd57a8b665fbb5b1d566c0b800dc6ede509e8e/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2340
                # hidden_states = self.projector(hidden_states)
                if a is None:
                    out = last_hidden_state.mean(dim=1)
                else:
                    padding_mask = (
                        self.feature_extractor._get_feature_vector_attention_mask(
                            last_hidden_state.shape[1], a
                        )
                    )
                    last_hidden_state[~padding_mask] = 0.0
                    out = last_hidden_state.sum(dim=1) / padding_mask.sum(dim=1).view(
                        -1, 1
                    )

            elif "xlm" in self.model_id:
                # Source:
                # https://github.com/huggingface/transformers/blob/7bbc62474391aff64f63fcc064c975752d1fa4de/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1306
                # https://github.com/huggingface/transformers/blob/7bbc62474391aff64f63fcc064c975752d1fa4de/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1573
                out = last_hidden_state[:, 0, :]
            elif "aya" in self.model_id:
                # Source:
                # https://github.com/huggingface/transformers/blob/ccbd57a8b665fbb5b1d566c0b800dc6ede509e8e/src/transformers/models/t5/modeling_t5.py#L2239
                eos_mask = x.eq(self.feature_extractor.config.eos_token_id).to(
                    last_hidden_state.device
                )
                batch_size, _, hidden_size = last_hidden_state.shape
                out = last_hidden_state[eos_mask, :].view(batch_size, -1, hidden_size)[
                    :, -1, :
                ]
            else:
                raise ValueError(f"Unknown model ID: {self.model_id}")

        # Final output shape: B x emb_dim
        return out

    def get_hidden_states(self, x, a=None):
        _, *hidden_state_list = self.feature_extractor(
            x, attention_mask=a, output_hidden_states=True
        ).hidden_states

        # Audio: B x n_layers X n_chunks x emb_dim
        # Text: B x n_layers x n_tokens x emb_dim
        hidden_states = torch.stack(hidden_state_list, dim=1)

        return hidden_states

    def get_attentions(self, x, a=None):
        # B x n_layers x n_heads x n_chunks x n_chunks
        attentions = torch.stack(
            self.feature_extractor(
                x, attention_mask=a, output_attentions=True
            ).attentions,
            dim=1,
        )

        return attentions


def _load_finetuned_xlsr(
    model,
    file="/home/common/speech_phylo/models/xlsr_300m_voxlingua107_ft.pt",
):
    """Load a finetuned Wav2Vec2Model for speech feature extraction.

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    file : str, optional
        Path to fine-tuned checkpoint
        (from https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/xlsr/README.md)

    Returns
    -------
    model_ : Wav2Vec2Model
        The loaded Wav2Vec2Model for speech feature extraction.
    """
    state_dict = torch.load(file)

    tmp = {
        k.replace("w2v_encoder.w2v_model.", "")
        .replace("mask_emb", "masked_spec_embed")
        .replace(".0.weight", ".conv.weight")
        .replace(".0.bias", ".conv.bias")
        .replace("fc1", "feed_forward.intermediate_dense")
        .replace("fc2", "feed_forward.output_dense")
        .replace("self_attn", "attention")
        .replace("attention_layer_norm", "layer_norm")
        .replace(".2.1", ".layer_norm")
        .replace("post_extract_proj", "feature_projection.projection")
        .replace("pos_conv", "pos_conv_embed")
        .replace("embed.conv.weight", "embed.conv.parametrizations.weight")
        .replace("weight_g", "weight.original0")
        .replace("weight_v", "weight.original1"): v
        for (k, v) in state_dict["model"].items()
    }

    tmp["feature_projection.layer_norm.bias"] = tmp.pop("layer_norm.bias")
    tmp["feature_projection.layer_norm.weight"] = tmp.pop("layer_norm.weight")

    missing_keys, unexpected_keys = model.load_state_dict(tmp, strict=False)

    print(f"missing keys: {missing_keys}\n" f"unexpec keys: {unexpected_keys}")

    return model


TRANSFORMERS_AUDIO_MODELS = {
    k: {
        "extractor": TransformersFeatureExtractor,
        "processor": TransformersAudioProcessor,
    }
    for k in (
        "facebook/wav2vec2-xls-r-300m",
        "facebook/mms-lid-126",
        "facebook/mms-lid-256",
        "facebook/mms-lid-4017",
    )
}
