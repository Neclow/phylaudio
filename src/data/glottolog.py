import json

import pandas as pd
from pyglottolog import Glottolog
from tqdm import tqdm

from src._config import DEFAULT_METADATA_DIR


def get_languoid_data(repos, glottocodes):
    g = Glottolog(repos)
    data = {}
    for glottocode in tqdm(
        glottocodes, total=len(glottocodes), desc="Fetching Glottolog data"
    ):
        if glottocode is not None:
            data[glottocode] = _get_languoid_data_single(g, glottocode)
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.merge(
        pd.DataFrame(df.codes.to_list(), index=df.index).add_prefix("H"),
        left_index=True,
        right_index=True,
    ).drop("codes", axis=1)
    return df


def _get_languoid_data_single(g, glottocode):
    languoid = g.languoid(id_=glottocode)
    # If the parsed language is a macrolanguage, average the lat/lon of its children
    if languoid.level.ordinal < 2:
        lats = []
        lons = []
        for child in languoid.children:
            if child.latitude is not None:
                lats.append(child.latitude)
            if child.longitude is not None:
                lons.append(child.longitude)
        if lats:
            languoid.latitude = sum(lats) / len(lats)
        if lons:
            languoid.longitude = sum(lons) / len(lons)
    lineage_codes = [x[1] for x in languoid.lineage]

    return {
        "name": languoid.name,
        "level": languoid.level.id,
        "glottocode": glottocode,
        "longitude": languoid.longitude,
        "latitude": languoid.latitude,
        "codes": lineage_codes,
    }


# pylint: disable=unused-argument
def filter_languages(
    dataset, glottocode, min_speakers=0, exclude=None, speaker_db="linguameta"
):
    exclude = set(exclude or [])

    with open(
        f"{DEFAULT_METADATA_DIR}/{dataset}/languages.json",
        "r",
        encoding="utf-8",
    ) as f:
        languages = json.load(f)

    enough_speakers = {}
    for k, v in languages.items():
        if k in exclude:
            continue
        n_speakers_db = v["speakers"][speaker_db]
        if pd.isna(n_speakers_db):
            n_speakers_db = max(v["speakers"].values())
        if n_speakers_db >= min_speakers:
            enough_speakers[k] = v["glottolog"]

    glottolog_path = f"{DEFAULT_METADATA_DIR}/{dataset}/glottolog.csv"

    lang_clf_df = (
        pd.read_csv(glottolog_path)
        .query("glottocode in @enough_speakers.values()")
        .set_index("glottocode")
        .ffill(axis=1)
        .astype(str)
    )

    stack_df = lang_clf_df.stack().to_frame(name="codes")

    filtered_stack = stack_df.query("codes.str.contains(@glottocode)")

    filtered_glottocodes = filtered_stack.unstack().index.to_list()

    assert len(filtered_glottocodes) > 0, "No languages found"

    filtered_languages = {
        k: v
        for k, v in languages.items()
        if v["glottolog"] in filtered_glottocodes and k not in exclude
    }

    return filtered_languages


# pylint: enable=unused-argument


def read_exclude_file(path):
    """Read a list of FLEURS dirs to exclude from a text file.

    One entry per line; '#' starts a comment; blank lines ignored.
    Returns [] when ``path`` is None.
    """
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        entries = []
        for line in f:
            entry = line.split("#", 1)[0].strip()
            if entry:
                entries.append(entry)
    return entries


def add_language_filter_args(parser):
    """Add the shared language-filter CLI flags to ``parser``.

    Wraps the three knobs that select the analysis language set:
    ``--glottocode``, ``--min-speakers``, ``--exclude-languages-file``.
    """
    parser.add_argument(
        "--glottocode",
        type=str,
        default="indo1319",
        help="Glottocode to filter languages",
    )
    parser.add_argument(
        "--min-speakers",
        type=float,
        default=0.0,
        help="Minimum number of speakers per language",
    )
    parser.add_argument(
        "--exclude-languages-file",
        type=str,
        default=None,
        help=(
            "Path to a text file listing FLEURS dirs to exclude "
            "(one per line, '#' comments allowed)"
        ),
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["male", "female"],
        help="Keep only samples of this gender (case-insensitive). Default: no filter.",
    )
    return parser


def get_language_to_family_mapping(glottolog_path, taxonomic_rank):
    lang_clf_df = pd.read_csv(glottolog_path).ffill(axis=1).astype(str)

    mapping = dict(zip(lang_clf_df.code, lang_clf_df[f"H{taxonomic_rank}"]))

    return mapping
