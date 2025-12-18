from typing import Final

from .audio import AudioProcessor, LogMelSpectrogramFeatureExtractor
from .baseline import BaselineFeatureExtractor
from .nemo import NeMoFeatureExtractor
from .opensmile import openSmileFeatureExtractor
from .speechbrain import SpeechbrainFeatureExtractor, SpeechbrainProcessor
from .transformers import TransformersAudioProcessor, TransformersFeatureExtractor
from .whisper import WhisperFeatureExtractor, WhisperProcessor

AUDIO_MAX_LENGTH: Final = 160000

OPENSMILE_MODELS: Final = {
    k: {
        "extractor": openSmileFeatureExtractor,
        "processor": AudioProcessor,
        "type": None,
        "arch": None,
    }
    for k in (
        "openSMILE/ComParE_2016",
        "openSMILE/eGeMAPSv02",
        "openSMILE/GeMAPSv01b",
    )
}

TRANSFORMERS_AUDIO_MODELS: Final = {
    k: {
        "extractor": TransformersFeatureExtractor,
        "processor": TransformersAudioProcessor,
        "type": "transformer",
        "arch": "wav2vec2",
    }
    for k in (
        "facebook/wav2vec2-xls-r-300m",
        "facebook/mms-lid-126",
        "facebook/mms-lid-256",
        "facebook/mms-lid-4017",
        "facebook/mms-1b-all",
        "mms-meta/mms-zeroshot-300m",
    )
}

WHISPER_MODELS: Final = {
    k: {
        "extractor": WhisperFeatureExtractor,
        "processor": WhisperProcessor,
        "type": "transformer",
        "arch": "Whisper",
    }
    for k in (
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3-turbo",
    )
}

BASELINE_MODELS: Final = {
    k: {
        "extractor": BaselineFeatureExtractor,
        "processor": AudioProcessor,
        "type": "conv2d",
        "arch": "Other (CNN)",
    }
    for k in (
        "baseline/CNN6",
        "baseline/CNN10",
    )
}

AUDIO_MODELS: Final = {
    "lms": {
        "extractor": LogMelSpectrogramFeatureExtractor,
        "processor": WhisperProcessor,
        "type": None,
        "arch": None,
    },
    "NeMo_ambernet": {
        "extractor": NeMoFeatureExtractor,
        "processor": AudioProcessor,
        "type": "conv1d",
        "arch": "Other (CNN)",
    },
    "speechbrain/lang-id-voxlingua107-ecapa": {
        "extractor": SpeechbrainFeatureExtractor,
        "processor": SpeechbrainProcessor,
        "type": "conv1d",
        "arch": "Other (CNN)",
    },
    **BASELINE_MODELS,
    **OPENSMILE_MODELS,
    **TRANSFORMERS_AUDIO_MODELS,
    **WHISPER_MODELS,
}


MODEL_ZOO: Final = {
    **{
        i: {**vi, "dtype": "audio", "max_length": AUDIO_MAX_LENGTH}
        for i, vi in AUDIO_MODELS.items()
    }
}
