"""Annotate FLEURS sentences with spaCy POS tags and named entities.

Produces ``data/metadata/{dataset}/spacy.csv`` with per-sentence flags for
proper nouns and 11 NER entity types, POS token counts, and a Heylighen &
Dewaele (1999) formality F-score.

Only English (en_core_web_sm) is used.  English covers 100% of the FLEURS
dev+test sentence universe and 98% of all sentences.  For full multilingual
coverage (including the ~33 train-only gaps), consider Stanza
(https://stanfordnlp.github.io/stanza/) which provides per-language POS and
NER models for 91+ languages.

Usage::

    pixi run spacy-annotate          # defaults to fleurs-r
    pixi run spacy-annotate fleurs   # or specify dataset
"""

import argparse

import pandas as pd
import spacy
from pathlib import Path

from src._config import DEFAULT_AUDIO_DIR, DEFAULT_METADATA_DIR

NER_TYPES = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
]

POS_FORMALITY = {
    "NOUN": "n_noun",
    "ADJ": "n_adj",
    "ADP": "n_adp",
    "DET": "n_det",
    "PRON": "n_pron",
    "VERB": "n_verb",
    "ADV": "n_adv",
    "INTJ": "n_intj",
}


def f_score(counts: dict[str, int], n_total: int) -> float:
    """Heylighen & Dewaele (1999) formality F-score.

    F = (noun% + adj% + adp% + det% - pron% - verb% - adv% - intj% + 100) / 2
    """
    if n_total == 0:
        return 50.0
    pct = {k: 100.0 * v / n_total for k, v in counts.items()}
    return (
        pct.get("n_noun", 0) + pct.get("n_adj", 0)
        + pct.get("n_adp", 0) + pct.get("n_det", 0)
        - pct.get("n_pron", 0) - pct.get("n_verb", 0)
        - pct.get("n_adv", 0) - pct.get("n_intj", 0)
        + 100
    ) / 2


def annotate_sentence(doc) -> dict:
    """Extract NLP annotations from a spaCy Doc."""
    row = {}

    row["propn"] = int(any(tok.pos_ == "PROPN" for tok in doc))

    ent_labels = {ent.label_ for ent in doc.ents}
    for ner_type in NER_TYPES:
        row[ner_type] = int(ner_type in ent_labels)

    counts = {}
    for tok in doc:
        col = POS_FORMALITY.get(tok.pos_)
        if col:
            counts[col] = counts.get(col, 0) + 1
    n_total = len(doc)
    for col in POS_FORMALITY.values():
        row[col] = counts.get(col, 0)
    row["n_total"] = n_total
    row["f_score"] = f_score(counts, n_total)

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", default="fleurs-r", help="Dataset name (default: fleurs-r)")
    args = parser.parse_args()

    # FLEURS-R reuses the same sentence universe as FLEURS, so we read the
    # English text from whichever copy exists (preferring the original FLEURS).
    fleurs_dir = Path(DEFAULT_AUDIO_DIR) / "fleurs" / "en_us" / "en_us"
    if not fleurs_dir.exists():
        fleurs_dir = Path(DEFAULT_AUDIO_DIR) / args.dataset / "en_us" / "en_us"
    if not fleurs_dir.exists():
        raise FileNotFoundError(f"English FLEURS directory not found: {fleurs_dir}")
    print(f"Reading English sentences from: {fleurs_dir}")

    nlp = spacy.load("en_core_web_sm")
    print(f"Loaded spaCy model: {nlp.meta['name']}")

    rows = []
    for split in ("dev", "test", "train"):
        tsv_path = fleurs_dir / f"{split}.tsv"
        df = pd.read_csv(
            tsv_path, sep="\t", header=None,
            names=["id", "file", "text", "norm", "chars", "samples", "gender"],
        )
        sents = df.drop_duplicates("id")[["id", "text"]]
        print(f"{split}: {len(sents)} unique sentences")

        texts = sents["text"].tolist()
        ids = sents["id"].tolist()

        for doc, sid in zip(nlp.pipe(texts, batch_size=64), ids):
            row = annotate_sentence(doc)
            row["split_id"] = f"{split}_{sid}"
            rows.append(row)

    result = pd.DataFrame(rows).set_index("split_id")

    out_path = Path(DEFAULT_METADATA_DIR) / args.dataset / "spacy.csv"
    result.to_csv(out_path, float_format="%.4f")

    n_clean = (result[["propn"] + NER_TYPES].sum(axis=1) == 0).sum()
    print(f"\nSaved {out_path} ({len(result)} sentences, {n_clean} pass NER filter)")


if __name__ == "__main__":
    main()
