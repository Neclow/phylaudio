import json
import multiprocessing
import os
import textwrap
import warnings
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path

import pandas as pd
import rpy2.robjects as ro
import torch
from ete3 import Tree
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from tqdm import tqdm

from ..._config import DEFAULT_METADATA_DIR, DEFAULT_PER_SENTENCE_DIR, MIN_LANGUAGES
from ...utils import _count_file_lines
from .newick import apply_language_mapping_to_newick
from .tree import (
    make_distance_tree,
    make_iqtree_tree,
    make_parsimony_tree,
    make_raxml_tree,
)

DISCRETE_METHODS = {
    "iqtree2": {
        "tree_ext": "treefile",
        "func": make_iqtree_tree,
    },
    "raxml": {
        "tree_ext": "raxml.bestTree",
        "func": make_raxml_tree,
    },
    "pratchet": {
        "tree_ext": "nwk",
        "func": make_parsimony_tree,
    },
}


PDIST_METHODS = ("fastme", "nj", "upgma")


class PhyloWriter(ABC):
    def __init__(self, run_id, dtype, data_ext, **kwargs):
        self.run_id = run_id

        self.data_ext = data_ext

        run_dir = f"{DEFAULT_PER_SENTENCE_DIR}/{dtype}/{run_id}"

        with open(f"{run_dir}/cfg.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        with open(
            f"{DEFAULT_METADATA_DIR}/{cfg['dataset']}/languages.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.languages = json.load(f)

        self.input_files = glob(f"{run_dir}/*.{data_ext}")

        self.tree_method = cfg["method"]

        self.tree_threads = cfg["n_threads"]

        self.output_file = f"{run_dir}/_trees.nex"

        self.started = False
        self.ended = False

    def start(self):
        """Starter for a nexus file"""
        full_language_names = " ".join(
            sorted([v["full"].replace(" ", "_") for v in self.languages.values()])
        )

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    f"""\
                    #NEXUS
                    Begin TAXA;
                    Dimensions ntax={len(self.languages)};
                    TaxLabels {full_language_names};
                    End;

                    BEGIN TREES;
                    """
                )
            )

        self.started = True

    def check_can_write(self):
        assert self.started and not self.ended, (
            "Nexus file not started or ended. "
            f"Got started = {self.started}, ended = {self.ended}."
        )

    @abstractmethod
    def process(self, file):
        raise NotImplementedError

    def write(self, export_trees=True):
        print(
            f"(phylo): Running {self.tree_method} with {self.tree_threads} thread(s)..."
        )

        # Start writing to nexus
        self.start()

        # Process all files
        if self.tree_threads > 1:
            with multiprocessing.Pool(processes=self.tree_threads) as pool:
                work = pool.imap_unordered(
                    self.process,
                    self.input_files,
                    chunksize=max(1, len(self.input_files) // (self.tree_threads * 2)),
                )

                outputs = list(tqdm(work, total=len(self.input_files)))
        else:
            outputs = [self.process(file) for file in tqdm(self.input_files)]

        # Write output trees to nexus
        skipped = []
        with open(self.output_file, "a", encoding="utf-8") as f:
            for output in outputs:
                match output:
                    case str(newick_line):
                        f.write(newick_line)
                    case Exception(args=(skipped_,)):
                        skipped.append(skipped_)
            f.write("END;")

        # Nexus file is written
        self.ended = True

        if len(skipped) > 0:
            warnings.warn(
                f"{len(skipped)} warnings encountered: {skipped}", UserWarning
            )

        # Export trees to a txt file (without nexus boilerplate)
        if export_trees:
            nexus_to_txt(self.output_file)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            + textwrap.indent(
                text=(
                    f"run_id='{self.run_id}',\n"
                    f"input_files={len(self.input_files)} .{self.data_ext} files (not shown),\n"
                    f"tree_method='{self.tree_method}',\n"
                    f"output_file='{self.output_file}'\n"
                ),
                prefix="  ",
            )
            + ")"
        )


class DiscretePhyloWriter(PhyloWriter):
    def __init__(self, run_id, dtype="discrete", **kwargs):
        super().__init__(run_id, dtype=dtype, data_ext="fa")

        if self.tree_method in DISCRETE_METHODS:
            self.tree_ext = DISCRETE_METHODS[self.tree_method]["tree_ext"]
            self.func = DISCRETE_METHODS[self.tree_method]["func"]
        else:
            raise ValueError(
                f"Unknown discrete tree inference method: `{self.tree_method}`."
            )

        self.kwargs = kwargs

    def process(self, file):
        self.check_can_write()
        stem = Path(file).stem
        try:
            if _count_file_lines(file) < 2 * MIN_LANGUAGES:
                return ValueError(f"{stem}: Less than {MIN_LANGUAGES} sequences")

            self.func(file, **self.kwargs)

            newick = apply_language_mapping_to_newick(
                newick_file=f"{file}.{self.tree_ext}",
                languages=self.languages,
                key="full",
            )
            return f"\tTree {Path(file).stem} = {newick}\n"
        except RRuntimeError as err:
            return ValueError(f"{stem}: {str(err)}")


class DistancePhyloWriter(PhyloWriter):
    def __init__(self, run_id, layer=-1):
        super().__init__(run_id, dtype="pdist", data_ext="pt")

        assert self.tree_method in (
            PDIST_METHODS
        ), f"(pdist) Unknown inference method: `{self.tree_method}`."

        self.layer = layer

        self.label2language = dict(
            enumerate([v["full"] for v in self.languages.values()])
        )

    def process(self, file):
        self.check_can_write()

        stem = Path(file).stem
        data = torch.load(file)

        # Average distances by counts
        distance_matrix = data[self.layer]["distance_matrix"]
        count_matrix = data[self.layer]["count_matrix"]
        distance_matrix_norm = (
            distance_matrix.cpu().div(count_matrix.cpu().clip(min=1)).numpy()
        )

        distance_df = pd.DataFrame(distance_matrix_norm).rename(
            index=self.label2language, columns=self.label2language
        )

        # Drop all-zeros rows & cols
        any_nnz_idxs = (distance_df != 0).any(axis=1)
        distance_df = distance_df.loc[any_nnz_idxs, any_nnz_idxs]

        if distance_df.shape[0] < MIN_LANGUAGES:
            return ValueError(f"{stem}: Less than {MIN_LANGUAGES} sequences")

        newick = make_distance_tree(distance_df, method=self.tree_method).replace(
            "_", " "
        )

        return f"\tTree {stem} = {newick}\n"


def nexus_to_txt(nexus_file):
    """Extract trees from a nexus file and save to a txt file"""

    txt_file = f"{os.path.splitext(nexus_file)[0]}.txt"

    with localconverter(ro.default_converter):
        importr("ape")

        ro.globalenv["nexus_file"] = nexus_file
        ro.globalenv["txt_file"] = txt_file

        ro.r(
            """
            tr <- read.nexus(nexus_file)
            write.tree(tr, file=txt_file)
            """
        )


def nexus_to_densitree(nexus_file, root=None):
    """Produce a densitree plot from a nexus file"""
    stem = os.path.splitext(nexus_file)[0]

    def _prepare_trees(stem):
        # Densitree works best with rooted trees
        output_file = f"{stem}_rooted.nex"

        add_line = False

        lines_to_write = []

        max_leaves = 0

        index = 0

        with open(nexus_file, "r", encoding="utf-8") as f_in:
            for i, line in enumerate(f_in.readlines()):
                if line.strip() == "END;" and add_line:
                    lines_to_write.append(line)
                elif add_line:
                    meta, newick = line.split(" = ")

                    tree = Tree(newick.strip())

                    leaves = tree.get_leaf_names()

                    # Making the tree rooted and ultrametric
                    if root is not None and root in leaves:
                        outgroup = root
                    else:
                        outgroup = tree.get_midpoint_outgroup()

                    tree.set_outgroup(outgroup)
                    tree.convert_to_ultrametric()

                    line_clean = f"{meta} = {tree.write(format=1)}\n"

                    n_leaves = len(leaves)

                    if n_leaves > max_leaves:
                        max_leaves = n_leaves
                        lines_to_write.insert(index, line_clean)
                    else:
                        lines_to_write.append(line_clean)
                else:
                    lines_to_write.append(line)

                if line.strip() == "BEGIN TREES;":
                    add_line = True
                    index = i + 1

        with open(output_file, "w", encoding="utf-8") as f_out:
            for line in lines_to_write:
                f_out.write(line)

        consensus_file = f"{stem}_astral4.txt"

        consensus_tree = Tree(consensus_file)

        consensus_tree.set_outgroup(root)
        consensus_tree.convert_to_ultrametric()

        consensus_tree.write(
            format=1, outfile=f"{os.path.splitext(consensus_file[0])}_rooted.txt"
        )

        return output_file

    clean_nexus_file = _prepare_trees(stem)

    with localconverter(ro.default_converter):
        importr("phangorn")

        ro.globalenv["nexus_file"] = clean_nexus_file
        ro.globalenv["consensus"] = f"{stem}_astral4_rooted.txt"
        ro.globalenv["output_file"] = f"{stem}_densitree.pdf"
        ro.globalenv["root_"] = root

        ro.r(
            """
            trees <- read.nexus(nexus_file)
            c_tree <- read.tree(consensus)

            pdf(output_file)
            densiTree(
                trees,
                consensus = root(c_tree, root_),
                alpha = 0.01,
                width = 0.1,
                cex = 0.5,
                jitter = list(amount=0.1, random=TRUE),
                col = "#12711c",
                font = 2
            )
            dev.off()
            """
        )
