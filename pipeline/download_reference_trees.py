# pylint: disable=redefined-outer-name
"""Download and process reference phylogenetic trees."""

import json
import os
from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from io import StringIO

import pandas as pd
import pyglottolog
import requests
from Bio.Phylo import NewickIO, NexusIO
from ete3 import Tree

from src._config import (
    DEFAULT_BEAST_DIR,
    DEFAULT_METADATA_DIR,
    DEFAULT_METADATA_KEY,
    DEFAULT_REFERENCE_TREE_PROCESSED_DIR,
    DEFAULT_REFERENCE_TREE_RAW_DIR,
)
from src.data.glottolog import filter_languages_from_glottocode


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

        # Find language matches: glottocode -> key name
        matches = []
        mapping = {}
        for language_data in languages_to_prune.values():
            glottocode = language_data["glottolog"]
            if len(tree.search_nodes(name=glottocode)) > 0:
                matches.append(glottocode)
            mapping[glottocode] = language_data[DEFAULT_METADATA_KEY]

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

            # Build glottocode -> key name mapping
            glottocode2key = {
                v["glottolog"]: v[DEFAULT_METADATA_KEY]
                for v in languages_to_prune.values()
            }

            # Prune and rename leaves (asjp name -> glottocode -> key name)
            tree.prune(
                list(matches.keys()), preserve_branch_length=preserve_branch_length
            )
            for leaf in tree.iter_leaves():
                glottocode = matches[leaf.name]
                leaf.name = glottocode2key[glottocode]

            diff = set(glottocodes_to_prune) - set(matches.values())
            absent_key_languages = [
                v[DEFAULT_METADATA_KEY]
                for v in languages_to_prune.values()
                if v["glottolog"] in diff
            ]
            # Absent in Jager tree: Kabuverdianu, Sindhi
            print(f"Absent languages in {self.name} tree: {absent_key_languages}")

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

            # Build glottocode -> key name mapping
            glottocode2key = {
                v["glottolog"]: v[DEFAULT_METADATA_KEY]
                for v in languages_to_prune.values()
            }

            # NOTE Absent in GLED tree: Kabuverdianu, Sindhi
            diff = set(glottocodes_to_prune) - set(matches.values())
            absent_key_languages = [
                v[DEFAULT_METADATA_KEY]
                for v in languages_to_prune.values()
                if v["glottolog"] in diff
            ]
            print(f"Absent languages in {self.name} tree: {absent_key_languages}")

            # Prune and rename leaves (gled name -> glottocode -> key name)
            tree.prune(
                list(matches.keys()), preserve_branch_length=preserve_branch_length
            )
            for leaf in tree.iter_leaves():
                glottocode = matches[leaf.name]
                leaf.name = glottocode2key[glottocode]

            self.write(tree, process_args)


class IecorTreeProcessor(URLTreeProcessor):
    def process(
        self,
        languages_to_prune,
        process_args,
        preserve_branch_length=True,
    ):
        tree = self.parse(self.raw_file)

        # Create mapping from tree leaf names to key language names
        mapping = {
            v.get(self.name, None): v[DEFAULT_METADATA_KEY]
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
            key_names = [v[DEFAULT_METADATA_KEY] for v in languages_to_prune.values()]
            leaf_names = tree.get_leaf_names()
            print(
                f"Absent languages in {self.name} tree: {sorted(set(key_names) - set(leaf_names))}"
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

EXTRA_IECOR_TREES = {
    "posterior": {
        "url": "https://share.eva.mpg.de/public.php/dav/files/E4Am2bbBA3qLngC/01_Main_Analysis_M3/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined.trees",
        "file": f"{DEFAULT_BEAST_DIR}/iecor/raw.trees",
    },
    "prior": {
        "url": "https://share.eva.mpg.de/public.php/dav/files/E4Am2bbBA3qLngC/01_Main_Analysis_M3/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined_PRIOR.trees",
        "file": f"{DEFAULT_BEAST_DIR}/iecor/prior/raw.trees",
    },
}
# pylint: enable=line-too-long


def parse_args():
    parser = ArgumentParser(
        description="Arguments for Glottolog lineage extraction",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=(
            "Dataset. Example: `fleurs`. "
            f"Has to have a folder in `{DEFAULT_METADATA_DIR}` with a `languages.json` file."
        ),
    )
    parser.add_argument(
        "-g",
        "--glottolog-dir",
        default="extern/glottolog",
        help="Path to glottolog folder",
    )
    parser.add_argument(
        "--glottocode",
        type=str,
        default="indo1319",
        help="Glottolog languoid code for the root of the tree to extract",
    )
    parser.add_argument(
        "--min-speakers",
        type=float,
        default=0.0,
        help="Minimum number of speakers for a language to be included (in millions)",
    )
    parser.add_argument(
        "--preserve-branch-length",
        action="store_true",
        help="Whether to preserve branch lengths when pruning trees",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing downloaded files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get json language data
    languages_filtered = filter_languages_from_glottocode(
        dataset=args.dataset,
        glottocode=args.glottocode,
        min_speakers=args.min_speakers,
    )

    for key, processor_utils in REFERENCE_TREES.items():
        # Prepare tree processor
        ProcessorCls = processor_utils["downloader"]
        extra_args = {}
        if key == "glottolog":
            extra_args["glottolog_dir"] = args.glottolog_dir
            extra_args["glottocode"] = args.glottocode
        elif key.startswith("asjp"):
            extra_args["asjp_url"] = processor_utils["asjp_url"]
        tree_processor = ProcessorCls(
            name=key,
            url=processor_utils.get("url", None),
            ext=processor_utils["ext"],
            **extra_args,
        )

        # Download raw tree file
        tree_processor.maybe_download(overwrite=args.overwrite)

        # Process tree file to only include relevant languages
        print(f"Processing: {key}...")
        tree_processor.process(
            languages_to_prune=languages_filtered,
            process_args=vars(args),
            preserve_branch_length=args.preserve_branch_length,
        )

    for key, extra_iecor_info in EXTRA_IECOR_TREES.items():
        print(f"Downloading extra IECoR trees: {key}...")
        if os.path.exists(extra_iecor_info["file"]) and not args.overwrite:
            print(
                f"Target file {extra_iecor_info['file']} already exists. Skipping download."
            )
        else:
            response = requests.get(
                extra_iecor_info["url"], allow_redirects=True, timeout=10
            )
            response.raise_for_status()
            with open(extra_iecor_info["file"], "wb") as f:
                f.write(response.content)

    print("Downloaded reference trees.")
