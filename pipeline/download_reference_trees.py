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

from src._config import DEFAULT_METADATA_DIR
from src.data.glottolog import filter_languages_from_glottocode

RAW_DIR = "data/trees/references/raw"
PROCESSED_DIR = "data/trees/references/processed"


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

        self.raw_file = f"{RAW_DIR}/{self.name}.{self.ext}"
        self.processed_file = f"{PROCESSED_DIR}/{self.name}.nwk"
        self.processed_args_file = f"{PROCESSED_DIR}/{self.name}.json"

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
        # Write processed tree
        processed_tree.write(outfile=self.processed_file, format=1)

        # Write args used to process tree
        with open(self.processed_args_file, "w", encoding="utf-8") as f:
            json.dump(process_args, f, indent=4)


class GlottologTreeProcessor(BaseTreeProcessor):
    def __init__(self, url, ext, glottolog_dir, languoid):
        super().__init__(name="glottolog", url=url, ext=ext)

        self.glottolog_dir = glottolog_dir
        self.languoid = languoid

    def download(self):
        """Download the Glottolog tree for a specific languoid."""
        G = pyglottolog.Glottolog(self.glottolog_dir)
        indo1319 = G.languoid(self.languoid)
        node = indo1319.newick_node()
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
            glottocode = language_data[self.name]
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
    def __init__(self, name, url, ext):
        super().__init__(name=name, url=url, ext=ext)

    def download(self):
        """Download a tree file from the specified URL."""
        response = requests.get(self.url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        with open(self.raw_file, "wb") as f:
            f.write(response.content)


class GledTreeProcessor(URLTreeProcessor):
    def __init__(self, url, ext):
        super().__init__(name="gled", url=url, ext=ext)

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
        # Manual fix for Pashto
        matches["NorthernPashto_nort2646"] = "pash1269"

        # Print stats:
        # NOTE: Should have 1 mismatch: Sindhi absent in GLED tree
        diff = set(glottocodes_to_prune) - set(matches.values())
        absent_full_languages = [
            v["full"] for v in languages_to_prune.values() if v["glottolog"] in diff
        ]
        print(f"Absent languages in {self.name} tree: {absent_full_languages}")

        # Prune and rename leaves
        tree.prune(list(matches.keys()), preserve_branch_length=preserve_branch_length)
        for leaf in tree.iter_leaves():
            leaf.name = matches[leaf.name]

        self.write(tree, process_args)


class IecorTreeProcessor(URLTreeProcessor):
    def __init__(self, url, ext):
        super().__init__(name="iecor", url=url, ext=ext)

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

        # Prune and rename leaves
        tree.prune(list(mapping.keys()), preserve_branch_length=preserve_branch_length)
        for leaf in tree.iter_leaves():
            leaf.name = mapping[leaf.name]

        # Print absent languages
        # NOTE: should have the following absent languages:
        # {'Oriya', 'Sindhi', 'Serbian', 'Gujarati', 'Tajik', 'Afrikaans', 'Bosnian', 'Galician'}
        # Serbian is missing because it's represented as Serbo-Croatian in the tree (via Croatian)
        full_names = [v["full"] for v in languages_to_prune.values()]
        leaf_names = tree.get_leaf_names()
        print(
            f"Absent languages in {self.name} tree: {set(full_names) - set(leaf_names)}"
        )

        self.write(tree, process_args)


class AsjpTreeProcessor(URLTreeProcessor):
    def __init__(self, url, ext, asjp_url):
        super().__init__(name="asjp", url=url, ext=ext)

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

        # Oriya named as ORIYA_KOTIA in asjp_languages
        matches["IE.INDIC.ORIYA"] = "adiv1239"
        # Pashto has many entries, we pick Northern Pashto (most spoken)
        matches["IE.IRANIAN.NORTHERN_PASHTO"] = "pash1269"

        # Prune and rename leaves
        tree.prune(list(matches.keys()), preserve_branch_length=preserve_branch_length)
        for leaf in tree.iter_leaves():
            leaf.name = matches[leaf.name]

        # Print stats
        # NOTE: Should have 1 mismatch: Sindhi absent in Asjp tree
        diff = set(glottocodes_to_prune) - set(matches.values())
        absent_full_languages = [
            v["full"] for v in languages_to_prune.values() if v["glottolog"] in diff
        ]
        # Sindhi is absent in Jager018 tree
        print(f"Absent languages in {self.name} tree: {absent_full_languages}")

        self.write(tree, process_args)


REFERENCE_TREES = {
    "gled": {
        "url": "https://raw.githubusercontent.com/tresoldi/gled/refs/heads/main/releases/20221127/trees/indoeuropean.tree",
        "asjp_url": "https://raw.githubusercontent.com/lexibank/asjp/refs/tags/v21/cldf/languages.csv",
        "ext": "nwk",
        "downloader": GledTreeProcessor,
    },
    "glottolog": {"ext": "nwk", "downloader": GlottologTreeProcessor},
    "iecor": {
        "url": "https://share.eva.mpg.de/index.php/s/E4Am2bbBA3qLngC/download?path=%2F01_Main_Analysis_M3%2FIECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin&files=IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_mcc.tree",
        "ext": "nex",
        "downloader": IecorTreeProcessor,
    },
    "asjp": {
        "url": "https://osf.io/hgru8/download",
        "ext": "nwk",
        "downloader": AsjpTreeProcessor,
    },
}


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
        "--languoid",
        type=str,
        default="indo1319",
        help="Glottolog languoid code for the root of the tree to extract",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1.0,
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


def prepare_language_metadata(dataset, languoid, min_speakers):
    meta_dir = f"{DEFAULT_METADATA_DIR}/{dataset}"
    with open(f"{meta_dir}/languages.json", "r", encoding="utf-8") as f:
        languages = json.load(f)

    # Filter languages according to glottocode and min speakers
    glottocodes_filtered = filter_languages_from_glottocode(
        f"{meta_dir}/glottolog.csv",
        glottocode=languoid,
        min_speakers=min_speakers,
    )
    languages_filtered = {g: languages[g] for g in glottocodes_filtered}

    return languages_filtered


if __name__ == "__main__":
    args = parse_args()

    # Create directories for raw and processed trees
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Get json language data
    languages_filtered = prepare_language_metadata(
        dataset=args.dataset,
        languoid=args.languoid,
        min_speakers=args.min_speakers,
    )

    for key, processor_utils in REFERENCE_TREES.items():
        print(f"Processing: {key}...")

        # Prepare tree processor
        ProcessorCls = processor_utils["downloader"]
        extra_args = {}
        if key == "glottolog":
            extra_args["glottolog_dir"] = args.glottolog_dir
            extra_args["languoid"] = args.languoid
        elif key == "asjp":
            extra_args["asjp_url"] = processor_utils["asjp_url"]
        tree_processor = ProcessorCls(
            url=processor_utils.get("url", None),
            ext=processor_utils["ext"],
            **extra_args,
        )

        # Download raw tree file
        tree_processor.maybe_download(overwrite=args.overwrite)

        # Process tree file to only include relevant languages
        tree_processor.process(
            languages_to_prune=languages_filtered,
            process_args=vars(args),
            preserve_branch_length=args.preserve_branch_length,
        )

    print("Downloaded reference trees.")
