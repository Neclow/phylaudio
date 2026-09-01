"""Dataset loading, language filtering, and audio transforms.

Modules:
    datasets   — AudioDataset (per-utterance), FleursParallelDataset (per-sentence
                 grouped by language), EmbeddingDataset (pre-cached), load_dataset().
    glottolog  — Language filtering by glottocode, min speakers, gender; family mapping.
    speakerpop — Speaker population downloads (Wikimedia, LinguaMeta).
    nlp        — Sentence-level filters (NER-based proper noun removal).
    transforms — Audio transforms: VAD, Trim, Pad.
"""

from .datasets import AudioDataset, FleursParallelDataset, load_dataset
from .glottolog import (
    add_language_filter_args,
    filter_languages,
    get_language_to_family_mapping,
    read_exclude_file,
)
from .transforms import load_transforms

__all__ = [
    "AudioDataset",
    "FleursParallelDataset",
    "load_dataset",
    "add_language_filter_args",
    "filter_languages",
    "get_language_to_family_mapping",
    "read_exclude_file",
    "load_transforms",
]
