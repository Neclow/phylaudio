"""NLP-based sentence filtering for FLEURS datasets.

Uses spaCy English (en_core_web_sm) to annotate sentences with POS tags and
named entities.  This covers all dev+test sentences (English has 100% coverage
of the FLEURS dev+test sentence universe).  For train-only gaps or non-English
analysis, consider Stanza (https://stanfordnlp.github.io/stanza/), which
provides per-language POS/NER models for 91+ languages.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src._config import DEFAULT_METADATA_DIR

FLAG_COLUMNS = [
    "propn",
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
]


def load_sentence_filter(
    dataset: str, filter_type: Optional[str] = None
) -> Optional[set[str]]:
    """Return a set of split_ids that pass the given filter, or None.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. "fleurs-r").
    filter_type : str or None
        ``"ner"`` — keep only sentences with no proper nouns and no named
        entities (PROPN + 11 NER types).  ``None`` — no filtering.

    Returns
    -------
    set[str] or None
        Set of ``split_id`` strings (e.g. ``"dev_1544"``) that pass the
        filter, or ``None`` when *filter_type* is ``None``.
    """
    if filter_type is None:
        return None

    if filter_type != "ner":
        raise ValueError(f"Unknown filter_type: {filter_type!r} (expected 'ner' or None)")

    csv_path = Path(DEFAULT_METADATA_DIR) / dataset / "spacy.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `pixi run spacy-annotate` first."
        )

    df = pd.read_csv(csv_path, index_col="split_id")
    mask = df[FLAG_COLUMNS].sum(axis=1) == 0
    return set(df.index[mask])
