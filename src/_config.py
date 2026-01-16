from collections import OrderedDict
from typing import Final

import torch

RANDOM_STATE: Final = 42

SAMPLE_RATE: Final = 16000

NONE_TENSOR: Final = torch.tensor([-1])

# Default directories
DEFAULT_ROOT_DIR: Final = "data"
DEFAULT_AUDIO_DIR: Final = f"{DEFAULT_ROOT_DIR}/datasets"
DEFAULT_CACHE_DIR: Final = f"{DEFAULT_ROOT_DIR}/models"
DEFAULT_EVAL_DIR: Final = f"{DEFAULT_ROOT_DIR}/eval"
DEFAULT_METADATA_DIR: Final = f"{DEFAULT_ROOT_DIR}/metadata"
DEFAULT_TREE_DIR: Final = f"{DEFAULT_ROOT_DIR}/trees"
DEFAULT_BEAST_DIR: Final = f"{DEFAULT_TREE_DIR}/beast"
DEFAULT_PER_SENTENCE_DIR: Final = f"{DEFAULT_TREE_DIR}/per_sentence"

# Default filenames
DEFAULT_MERGED_FASTA_FILE: Final = "__merged.fa"
DEFAULT_MAPPED_FASTA_FILE: Final = "__merged_mapped.fa"
DEFAULT_SPLITSTREE_FASTA_FILE: Final = "__merged_splitstree.fa"

# Default number of threads for phylogenetic tree inference (iqtree/raxml)
DEFAULT_THREADS_TREE: Final = 4

# Default number for discrete to nexus
DEFAULT_THREADS_NEXUS: Final = 16

# Minimum number of different languages (leaves) to infer a tree
MIN_LANGUAGES: Final = 4

# Fleurs language names
# Copyright 2022 The Google and HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
_FLEURS_LANG_TO_ID: Final = OrderedDict(
    [
        ("Afrikaans", "af"),
        ("Amharic", "am"),
        ("Arabic", "ar"),
        ("Armenian", "hy"),
        ("Assamese", "as"),
        ("Asturian", "ast"),
        ("Azerbaijani", "az"),
        ("Belarusian", "be"),
        ("Bengali", "bn"),
        ("Bosnian", "bs"),
        ("Bulgarian", "bg"),
        ("Burmese", "my"),
        ("Catalan", "ca"),
        ("Cebuano", "ceb"),
        ("Mandarin Chinese", "cmn_hans"),
        ("Cantonese Chinese", "yue_hant"),
        ("Croatian", "hr"),
        ("Czech", "cs"),
        ("Danish", "da"),
        ("Dutch", "nl"),
        ("English", "en"),
        ("Estonian", "et"),
        ("Filipino", "fil"),
        ("Finnish", "fi"),
        ("French", "fr"),
        ("Fula", "ff"),
        ("Galician", "gl"),
        ("Ganda", "lg"),
        ("Georgian", "ka"),
        ("German", "de"),
        ("Greek", "el"),
        ("Gujarati", "gu"),
        ("Hausa", "ha"),
        ("Hebrew", "he"),
        ("Hindi", "hi"),
        ("Hungarian", "hu"),
        ("Icelandic", "is"),
        ("Igbo", "ig"),
        ("Indonesian", "id"),
        ("Irish", "ga"),
        ("Italian", "it"),
        ("Japanese", "ja"),
        ("Javanese", "jv"),
        ("Kabuverdianu", "kea"),
        ("Kamba", "kam"),
        ("Kannada", "kn"),
        ("Kazakh", "kk"),
        ("Khmer", "km"),
        ("Korean", "ko"),
        ("Kyrgyz", "ky"),
        ("Lao", "lo"),
        ("Latvian", "lv"),
        ("Lingala", "ln"),
        ("Lithuanian", "lt"),
        ("Luo", "luo"),
        ("Luxembourgish", "lb"),
        ("Macedonian", "mk"),
        ("Malay", "ms"),
        ("Malayalam", "ml"),
        ("Maltese", "mt"),
        ("Maori", "mi"),
        ("Marathi", "mr"),
        ("Mongolian", "mn"),
        ("Nepali", "ne"),
        ("Northern-Sotho", "nso"),
        ("Norwegian", "nb"),
        ("Nyanja", "ny"),
        ("Occitan", "oc"),
        ("Oriya", "or"),
        ("Oromo", "om"),
        ("Pashto", "ps"),
        ("Persian", "fa"),
        ("Polish", "pl"),
        ("Portuguese", "pt"),
        ("Punjabi", "pa"),
        ("Romanian", "ro"),
        ("Russian", "ru"),
        ("Serbian", "sr"),
        ("Shona", "sn"),
        ("Sindhi", "sd"),
        ("Slovak", "sk"),
        ("Slovenian", "sl"),
        ("Somali", "so"),
        ("Sorani-Kurdish", "ckb"),
        ("Spanish", "es"),
        ("Swahili", "sw"),
        ("Swedish", "sv"),
        ("Tajik", "tg"),
        ("Tamil", "ta"),
        ("Telugu", "te"),
        ("Thai", "th"),
        ("Turkish", "tr"),
        ("Ukrainian", "uk"),
        ("Umbundu", "umb"),
        ("Urdu", "ur"),
        ("Uzbek", "uz"),
        ("Vietnamese", "vi"),
        ("Welsh", "cy"),
        ("Wolof", "wo"),
        ("Xhosa", "xh"),
        ("Yoruba", "yo"),
        ("Zulu", "zu"),
    ]
)
_FLEURS_LANG_SHORT_TO_LONG: Final = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}
_FLEURS_LANG: Final = sorted(
    [
        "af_za",
        "am_et",
        "ar_eg",
        "as_in",
        "ast_es",
        "az_az",
        "be_by",
        "bn_in",
        "bs_ba",
        "ca_es",
        "ceb_ph",
        "cmn_hans_cn",
        "yue_hant_hk",
        "cs_cz",
        "cy_gb",
        "da_dk",
        "de_de",
        "el_gr",
        "en_us",
        "es_419",
        "et_ee",
        "fa_ir",
        "ff_sn",
        "fi_fi",
        "fil_ph",
        "fr_fr",
        "ga_ie",
        "gl_es",
        "gu_in",
        "ha_ng",
        "he_il",
        "hi_in",
        "hr_hr",
        "hu_hu",
        "hy_am",
        "id_id",
        "ig_ng",
        "is_is",
        "it_it",
        "ja_jp",
        "jv_id",
        "ka_ge",
        "kam_ke",
        "kea_cv",
        "kk_kz",
        "km_kh",
        "kn_in",
        "ko_kr",
        "ckb_iq",
        "ky_kg",
        "lb_lu",
        "lg_ug",
        "ln_cd",
        "lo_la",
        "lt_lt",
        "luo_ke",
        "lv_lv",
        "mi_nz",
        "mk_mk",
        "ml_in",
        "mn_mn",
        "mr_in",
        "ms_my",
        "mt_mt",
        "my_mm",
        "nb_no",
        "ne_np",
        "nl_nl",
        "nso_za",
        "ny_mw",
        "oc_fr",
        "om_et",
        "or_in",
        "pa_in",
        "pl_pl",
        "ps_af",
        "pt_br",
        "ro_ro",
        "ru_ru",
        "bg_bg",
        "sd_in",
        "sk_sk",
        "sl_si",
        "sn_zw",
        "so_so",
        "sr_rs",
        "sv_se",
        "sw_ke",
        "ta_in",
        "te_in",
        "tg_tj",
        "th_th",
        "tr_tr",
        "uk_ua",
        "umb_ao",
        "ur_pk",
        "uz_uz",
        "vi_vn",
        "wo_sn",
        "xh_za",
        "yo_ng",
        "zu_za",
    ]
)
_FLEURS_LONG_TO_LANG: Final = {
    _FLEURS_LANG_SHORT_TO_LONG["_".join(k.split("_")[:-1]) or k]: k
    for k in _FLEURS_LANG
}
_FLEURS_SHORT_TO_LANG = {
    v: _FLEURS_LONG_TO_LANG[k] for k, v in _FLEURS_LANG_TO_ID.items()
}
_FLEURS_NAMES: Final = {
    _FLEURS_LONG_TO_LANG[k]: {"fleurs": k, "iso639-1": v}
    for k, v in _FLEURS_LANG_TO_ID.items()
}

# Mapping from FLEURS directory to IE-CoR language name
FLEURS_TO_IECOR: Final = {
    "as_in": "Assamese",
    "be_by": "Belarusian",
    "bg_bg": "Bulgarian",
    "bn_in": "Bengali",
    "bs_ba": "SerboCroatian",
    "ca_es": "Catalan",
    "ckb_iq": "KurdishCJafi",
    "cs_cz": "Czech",
    "cy_gb": "WelshNorth",
    "da_dk": "Danish",
    "de_de": "German",
    "el_gr": "Greek",
    "en_us": "English",
    "es_419": "Spanish",
    "fa_ir": "PersianTehran",
    "fr_fr": "French",
    "ga_ie": "GaelicIrish",
    "gl_es": "Galician",
    "gu_in": "Gujarati",
    "hi_in": "Hindi",
    "hr_hr": "SerboCroatian",
    "hy_am": "ArmenianEastern",
    "is_is": "Icelandic",
    "it_it": "Italian",
    "lb_lu": "Luxembourgish",
    "lt_lt": "Lithuanian",
    "lv_lv": "Latvian",
    "mk_mk": "Macedonian",
    "mr_in": "Marathi",
    "nb_no": "NorwegianBokmal",
    "ne_np": "Nepali",
    "nl_nl": "Dutch",
    "or_in": "Oriya",
    "pa_in": "Punjabi",
    "pl_pl": "Polish",
    "ps_af": "Pashto",
    "pt_br": "Portuguese",
    "ro_ro": "Romanian",
    "ru_ru": "Russian",
    "sd_in": "Sindhi",
    "sk_sk": "Slovak",
    "sl_si": "Slovene",
    "sr_rs": "SerboCroatian",
    "sv_se": "Swedish",
    "tg_tj": "Tajik",
    "uk_ua": "Ukrainian",
    "ur_pk": "Urdu",
    "af_za": "Afrikaans",
}
