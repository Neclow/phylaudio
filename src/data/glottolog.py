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
    df = (
        df.merge(
            pd.DataFrame(df.codes.to_list(), index=df.index).add_prefix("H"),
            left_index=True,
            right_index=True,
        )
        .drop("codes", axis=1)
        .set_index("name")
    )
    return df


def _get_languoid_data_single(g, glottocode):
    languoid = g.languoid(id_=glottocode)
    lineage_codes = [x[1] for x in languoid.lineage]

    return {
        "name": languoid.name,
        "glottocode": glottocode,
        "longitude": languoid.longitude,
        "latitude": languoid.latitude,
        "codes": lineage_codes,
    }


# pylint: disable=unused-argument
def filter_languages_from_glottocode(
    filepath_or_dataset, glottocode, min_speakers=0, speaker_db="linguameta"
):
    with open(
        f"{DEFAULT_METADATA_DIR}/{filepath_or_dataset}/languages.json",
        "r",
        encoding="utf-8",
    ) as f:
        languages = json.load(f)

    enough_speakers = {
        k: v["glottolog"]
        for k, v in languages.items()
        if v["speakers"][speaker_db] >= min_speakers
    }

    if filepath_or_dataset.endswith(".csv"):
        glottolog_path = filepath_or_dataset
    else:
        glottolog_path = f"{DEFAULT_METADATA_DIR}/{filepath_or_dataset}/glottolog.csv"

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
        k: v for k, v in languages.items() if v["glottolog"] in filtered_glottocodes
    }

    return filtered_languages


# pylint: enable=unused-argument


def get_language_to_family_mapping(glottolog_path, taxonomic_rank):
    lang_clf_df = pd.read_csv(glottolog_path).ffill(axis=1).astype(str)

    mapping = dict(zip(lang_clf_df.code, lang_clf_df[f"H{taxonomic_rank}"]))

    return mapping
