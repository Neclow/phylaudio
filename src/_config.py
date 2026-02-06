import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from io import StringIO
from typing import Final

import pandas as pd
import pyglottolog
import requests
import torch
from Bio.Phylo import NewickIO, NexusIO
from ete3 import Tree

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
_FLEURS_TO_IECOR: Final = {
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
    "pa_in": "Punjabi",
    "pl_pl": "Polish",
    "ps_af": "Pashto",
    "pt_br": "Portuguese",
    "ro_ro": "Romanian",
    "ru_ru": "Russian",
    "sk_sk": "Slovak",
    "sl_si": "Slovene",
    "sr_rs": "SerboCroatian",
    "sv_se": "Swedish",
    "uk_ua": "Ukrainian",
    "ur_pk": "Urdu",
}

# Taxonset memberships by IECOR name (from BEAST2 template.xml)
_INDO1319_FAMILIES_TO_FLEURS = {
    "armenian": ["hy_am"],
    "baltic": ["lt_lt", "lv_lv"],
    "celtic": ["ga_ie", "cy_gb"],
    "germanic": [
        "af_za",
        "da_dk",
        "nl_nl",
        "en_us",
        "de_de",
        "is_is",
        "lb_lu",
        "nb_no",
        "sv_se",
    ],
    "greek": ["el_gr"],
    "indoaryan": [
        "as_in",
        "bn_in",
        "gu_in",
        "hi_in",
        "mr_in",
        "ne_np",
        "or_in",
        "pa_in",
        "sd_in",
        "ur_pk",
    ],
    "iranian": ["ckb_iq", "ps_af", "fa_ir", "tg_tj"],
    "romance": [
        "ast_es",
        "ca_es",
        "es_419",
        "fr_fr",
        "gl_es",
        "it_it",
        "kea_cv",
        "oc_fr",
        "pt_br",
        "ro_ro",
    ],
    "slavic": [
        "be_by",
        "bg_bg",
        "bs_ba",
        "cs_cz",
        "hr_hr",
        "mk_mk",
        "pl_pl",
        "ru_ru",
        "sr_rs",
        "sk_sk",
        "sl_si",
        "uk_ua",
    ],
}

_FLEURS_TO_INDO1319_FAMILIES = {
    fleurs_dir: family
    for family, dirs in _INDO1319_FAMILIES_TO_FLEURS.items()
    for fleurs_dir in dirs
}

DEFAULT_REFERENCE_TREE_DIR = "data/trees/references"
DEFAULT_REFERENCE_TREE_RAW_DIR = f"{DEFAULT_REFERENCE_TREE_DIR}/raw"
DEFAULT_REFERENCE_TREE_PROCESSED_DIR = f"{DEFAULT_REFERENCE_TREE_DIR}/processed"


class BaseTreeProcessor(ABC):
    """Base tree processor class

    This class defines the interface for downloading and processing phylogenetic trees.

    Parameters
    ----------
    name : str
        Name of the tree
    url : str
        URL to download the tree from
    ext : str
        File extension of the tree (e.g., 'nex', 'nwk')
    """

    def __init__(self, name, url, ext):
        self.name = name
        self.url = url
        self.ext = ext

        assert self.ext in ("nex", "nwk"), f"Unknown tree format: {self.ext}"

        # Create directories for raw and processed trees
        os.makedirs(DEFAULT_REFERENCE_TREE_RAW_DIR, exist_ok=True)
        os.makedirs(DEFAULT_REFERENCE_TREE_PROCESSED_DIR, exist_ok=True)

        self.raw_file = f"{DEFAULT_REFERENCE_TREE_RAW_DIR}/{self.name}.{self.ext}"
        self.processed_file = f"{DEFAULT_REFERENCE_TREE_PROCESSED_DIR}/{self.name}.nwk"
        self.processed_args_file = (
            f"{DEFAULT_REFERENCE_TREE_PROCESSED_DIR}/{self.name}.json"
        )

    def maybe_download(self, overwrite=False):
        """Maybe download the tree file.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the existing file, by default False
        """
        if os.path.exists(self.raw_file) and not overwrite:
            print(f"Target file {self.raw_file} already exists. Skipping download.")
        else:
            print(f"Downloading {self.name} tree...")
            self.download()

    @abstractmethod
    def download(self):
        """Download the tree file."""
        raise NotImplementedError

    def parse(self, content) -> Tree:
        """Parse the raw tree file into an ete3 Tree object.

        Parameters
        ----------
        content : str, file-like
            Path or content of the tree file.

        Returns
        -------
        Tree
            Parsed ete3 Tree object.
        """
        if self.ext == "nex":
            tree = next(iter(NexusIO.parse(content)))
            # Remove comments
            for clade in tree.find_clades():
                clade.comment = ""
            # Get Newick string
            newick = list(NewickIO.Writer([tree]).to_strings())[0]
        elif self.ext == "nwk":
            # Directly parse Newick
            newick = content
        else:
            raise ValueError(f"Unknown tree format: {self.ext}")
        tree = Tree(newick, format=1)
        return tree

    @abstractmethod
    def process(
        self,
        languages_to_prune,
        process_args,
        preserve_branch_length=True,
    ):
        """Process the tree file.

        Parameters
        ----------
        languages_to_prune : dict
            Dictionary of languages to prune from the raw tree.
        preserve_branch_length : bool, optional
            Whether to preserve branch lengths, by default True
        """
        raise NotImplementedError

    def write(self, processed_tree, process_args):
        """Write the processed tree and its processing arguments to files.

        Parameters
        ----------
        processed_tree : ete3.Tree
            Processed tree
        process_args : dict
            Processing arguments
        """
        if processed_tree.children:
            # Write processed tree
            processed_tree.write(outfile=self.processed_file, format=1)

            # Write args used to process tree
            with open(self.processed_args_file, "w", encoding="utf-8") as f:
                json.dump(process_args, f, indent=4)
        else:
            print(f"Processed tree for {self.name} is empty. Skipping write.")


class GlottologTreeProcessor(BaseTreeProcessor):
    def __init__(self, name, url, ext, glottolog_dir, glottocode):
        super().__init__(name=f"{name}_{glottocode}", url=url, ext=ext)

        self.glottolog_dir = glottolog_dir
        self.glottocode = glottocode

    def download(self):
        """Download the Glottolog tree for a specific languoid."""
        # pylint: disable=invalid-name
        G = pyglottolog.Glottolog(self.glottolog_dir)
        language_family = G.languoid(self.glottocode)
        # pylint: enable=invalid-name
        node = language_family.newick_node()
        tree = Tree(node.newick + ";", format=1)
        for n in tree.traverse():
            n.name = n.name.split("]")[0].split("[")[-1]
        tree.write(outfile=self.raw_file, format=1)

    def process(
        self,
        languages_to_prune,
        process_args,
        preserve_branch_length=True,
    ):
        tree = self.parse(self.raw_file)

        # Find language matches: glottocode -> full name
        matches = []
        mapping = {}
        for language_data in languages_to_prune.values():
            glottocode = language_data["glottolog"]
            if len(tree.search_nodes(name=glottocode)) > 0:
                matches.append(glottocode)
            mapping[glottocode] = language_data["full"]

        # Prune and rename leaves
        tree.prune(matches, preserve_branch_length=preserve_branch_length)
        for leaf in tree.iter_leaves():
            leaf.name = mapping[leaf.name]

        print(
            f"Absent languages in {self.name} tree: {set(mapping.keys()) - set(matches)}"
        )

        # Write processed tree
        self.write(tree, process_args)


class URLTreeProcessor(BaseTreeProcessor):
    def download(self):
        """Download a tree file from the specified URL."""
        response = requests.get(self.url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        with open(self.raw_file, "wb") as f:
            f.write(response.content)


class AsjpTreeProcessor(URLTreeProcessor):
    def __init__(self, name, url, ext, asjp_url):
        super().__init__(name=name, url=url, ext=ext)

        self.asjp_url = asjp_url

    def download_asjp_metadata(self):
        """Download ASJP language metadata from the ASJP GitHub repository."""
        response = requests.get(self.asjp_url, timeout=10)
        response.raise_for_status()
        # Transfer to DataFrame
        asjp_languages = pd.read_csv(
            StringIO(response.content.decode("utf-8")), sep=","
        )
        return asjp_languages

    def process(
        self,
        languages_to_prune,
        process_args,
        preserve_branch_length=True,
    ):
        tree = self.parse(self.raw_file)

        leaf_names = tree.get_leaf_names()
        glottocodes_to_prune = [v["glottolog"] for v in languages_to_prune.values()]

        # Download ASJP language metadata
        asjp_languages = self.download_asjp_metadata()

        # Find matches via ASJP names mapped to glottocodes
        asjp2glottocode = dict(zip(asjp_languages.Name, asjp_languages.Glottocode))
        matches = {}
        for k in leaf_names:
            asjp_name = k.split(".")[-1]
            glottolog_name = asjp2glottocode.get(asjp_name, None)
            if glottolog_name is not None:
                for fk in glottocodes_to_prune:
                    if glottolog_name == fk and fk not in matches.values():
                        matches[k] = fk
                        break

        if not matches:
            self.write(Tree(), process_args)
        else:
            if "indo1319" in self.name:
                # Oriya named as ORIYA_KOTIA in asjp_languages
                matches["IE.INDIC.ORIYA"] = "oriy1255"
                # Pashto has many entries, we pick Northern Pashto (most spoken)
                matches["IE.IRANIAN.NORTHERN_PASHTO"] = "nucl1276"
                # Irish named as IRISH_GAELIC in asjp_languages
                matches["IE.CELTIC.IRISH_GAELIC"] = "iris1253"
                # Serbian named as IE.SLAVIC.SERBOCROATIAN in asjp_languages
                matches["IE.SLAVIC.SERBOCROATIAN"] = "serb1264"
                # Latvian named as IE.BALTIC.LATVIAN in asjp_languages
                matches["IE.BALTIC.LATVIAN"] = "latv1249"

            # Prune and rename leaves
            tree.prune(
                list(matches.keys()), preserve_branch_length=preserve_branch_length
            )
            for leaf in tree.iter_leaves():
                leaf.name = matches[leaf.name]

            diff = set(glottocodes_to_prune) - set(matches.values())
            absent_full_languages = [
                v["full"] for v in languages_to_prune.values() if v["glottolog"] in diff
            ]
            # Absent in Jager tree: Kabuverdianu, Sindhi
            print(f"Absent languages in {self.name} tree: {absent_full_languages}")

            self.write(tree, process_args)


class GledTreeProcessor(URLTreeProcessor):
    def process(
        self,
        languages_to_prune,
        process_args,
        preserve_branch_length=True,
    ):
        tree = self.parse(self.raw_file)

        # Find language matches via glottocodes in gled leaf names
        leaf_names = tree.get_leaf_names()
        glottocodes_to_prune = [v["glottolog"] for v in languages_to_prune.values()]
        matches = {}
        for k in leaf_names:
            glottolog_name = k.split("_")[-1]
            for fk in glottocodes_to_prune:
                if glottolog_name == fk:
                    matches[k] = fk
                    break

        if not matches:
            self.write(Tree(), process_args)
        else:
            # Quick fix for Pashto, Latvian and Serbo-Croatian-Bosnian
            if "indo1319" in self.name:
                matches["NorthernPashto_nort2646"] = "nucl1276"
                matches["StandardLatvian_stan1325"] = "latv1249"
                matches["Serbian-Croatian-Bosnian_sout1528"] = "serb1264"

            # NOTE Absent in GLED tree: Kabuverdianu, Sindhi
            diff = set(glottocodes_to_prune) - set(matches.values())
            absent_full_languages = [
                v["full"] for v in languages_to_prune.values() if v["glottolog"] in diff
            ]
            print(f"Absent languages in {self.name} tree: {absent_full_languages}")

            # Prune and rename leaves
            tree.prune(
                list(matches.keys()), preserve_branch_length=preserve_branch_length
            )
            for leaf in tree.iter_leaves():
                leaf.name = matches[leaf.name]

            self.write(tree, process_args)


class IecorTreeProcessor(URLTreeProcessor):
    def process(
        self,
        languages_to_prune,
        process_args,
        preserve_branch_length=True,
    ):
        tree = self.parse(self.raw_file)

        # Create mapping from tree leaf names to full language names
        mapping = {
            v.get(self.name, None): v["full"]
            for v in reversed(languages_to_prune.values())
        }
        del mapping[None]

        if len(mapping) == 0:
            self.write(Tree(), process_args)
        else:
            # Prune and rename leaves
            tree.prune(
                list(mapping.keys()), preserve_branch_length=preserve_branch_length
            )
            for leaf in tree.iter_leaves():
                leaf.name = mapping[leaf.name]

            # Print absent languages
            # NOTE: should have the following absent languages:
            # {
            #   'Afrikaans', 'Asturian', 'Croatian', 'Galician', Gujarati',
            #   'Kabuverdianu', 'Occitan', 'Odia', 'Serbian', 'Sindhi', 'Tajik'
            # }
            # Croatian/Serbian are missing as represented (via Bosnian as BCMS)
            full_names = [v["full"] for v in languages_to_prune.values()]
            leaf_names = tree.get_leaf_names()
            print(
                f"Absent languages in {self.name} tree: {sorted(set(full_names) - set(leaf_names))}"
            )

            self.write(tree, process_args)


# pylint: disable=line-too-long
REFERENCE_TREES = {
    "gled_indo1319": {
        "url": "https://raw.githubusercontent.com/tresoldi/gled/refs/heads/main/releases/20221127/trees/indoeuropean.tree",
        "ext": "nwk",
        "downloader": GledTreeProcessor,
    },
    "gled_atla1278": {
        "url": "https://raw.githubusercontent.com/tresoldi/gled/bd1e2ff26a1332771c7550c1a4a3c3f26985369b/releases/20221127/bayesian/Atlantic-Congo.tree",
        "ext": "nwk",
        "downloader": GledTreeProcessor,
    },
    "glottolog": {"ext": "nwk", "downloader": GlottologTreeProcessor},
    "iecor": {
        "url": "https://share.eva.mpg.de/public.php/dav/files/E4Am2bbBA3qLngC/01_Main_Analysis_M3/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_mcc.tree",
        "ext": "nex",
        "downloader": IecorTreeProcessor,
    },
    "asjp_indo1319": {
        "url": "https://osf.io/hgru8/download",
        "asjp_url": "https://raw.githubusercontent.com/lexibank/asjp/refs/tags/v21/cldf/languages.csv",
        "ext": "nwk",
        "downloader": AsjpTreeProcessor,
    },
    "asjp_atla1278": {
        "url": "https://osf.io/wjp46/download",
        "asjp_url": "https://raw.githubusercontent.com/lexibank/asjp/refs/tags/v21/cldf/languages.csv",
        "ext": "nwk",
        "downloader": AsjpTreeProcessor,
    },
}
# pylint: enable=line-too-long
