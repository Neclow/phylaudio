import pandas as pd


# pylint: disable=unused-argument
def filter_languages_from_glottocode(glottolog_path, glottocode, min_speakers=0):
    lang_clf_df = (
        pd.read_csv(glottolog_path)
        .query("n_speakers >= @min_speakers")
        .set_index("code")
        .ffill(axis=1)
        .astype(str)
    )

    stack_df = lang_clf_df.stack().to_frame(name="codes")

    filtered_stack = stack_df.query("codes.str.contains(@glottocode)")

    filtered_languages = filtered_stack.unstack().index.to_list()

    assert len(filtered_languages) > 0, "No languages found"

    return filtered_languages


# pylint: enable=unused-argument


def get_language_to_family_mapping(glottolog_path, taxonomic_rank):
    lang_clf_df = pd.read_csv(glottolog_path).ffill(axis=1).astype(str)

    mapping = dict(zip(lang_clf_df.code, lang_clf_df[f"H{taxonomic_rank}"]))

    return mapping
