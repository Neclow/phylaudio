from typing import Final

from .audio import AudioProcessor, LogMelSpectrogramFeatureExtractor
from .nemo import NeMoFeatureExtractor
from .opensmile import openSmileFeatureExtractor
from .speechbrain import SpeechbrainFeatureExtractor, SpeechbrainProcessor
from .transformers import TransformersAudioProcessor, TransformersFeatureExtractor
from .whisper import WhisperFeatureExtractor, WhisperProcessor

AUDIO_MAX_LENGTH: Final = 160000

OPENSMILE_MODELS: Final = {
    k: {"extractor": openSmileFeatureExtractor, "processor": AudioProcessor}
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
    }
    for k in (
        "facebook/wav2vec2-xls-r-300m",
        "facebook/mms-lid-126",
        "facebook/mms-lid-256",
        "facebook/mms-lid-4017",
    )
}

WHISPER_MODELS: Final = {
    k: {"extractor": WhisperFeatureExtractor, "processor": WhisperProcessor}
    for k in (
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3-turbo",
    )
}

AUDIO_MODELS: Final = {
    "lms": {
        "extractor": LogMelSpectrogramFeatureExtractor,
        "processor": WhisperProcessor,
    },
    "NeMo_ambernet": {
        "extractor": NeMoFeatureExtractor,
        "processor": AudioProcessor,
    },
    "speechbrain/lang-id-voxlingua107-ecapa": {
        "extractor": SpeechbrainFeatureExtractor,
        "processor": SpeechbrainProcessor,
    },
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
