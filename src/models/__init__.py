"""Pre-trained audio feature extractors.

Modules:
    _base        — BaseFeatureExtractor / BaseProcessor abstract interfaces.
    _model_zoo   — MODEL_ZOO registry mapping model IDs to extractor/processor classes.
    audio        — Log-mel spectrogram extractor (shared by whisper/baseline).
    baseline     — CNN6/CNN10 classification baselines.
    embedding    — Pass-through extractor for pre-cached embeddings.
    nemo         — NVIDIA NeMo AmberNet speaker encoder.
    opensmile    — openSMILE handcrafted acoustic features (eGeMAPSv02).
    speechbrain  — SpeechBrain ECAPA-TDNN (VoxLingua107).
    transformers — HuggingFace models (XLS-R, MMS, HuBERT): mean-pooled hidden states.
    whisper      — OpenAI Whisper: encoder-decoder embeddings from log-mel spectrograms.
"""

from ._model_zoo import MODEL_ZOO

__all__ = ["MODEL_ZOO"]
