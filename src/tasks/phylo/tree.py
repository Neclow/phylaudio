"""Tree building functions"""

import linecache

import pandas as pd
import rpy2.rinterface_lib.callbacks
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from ..._config import DEFAULT_THREADS_TREE, RANDOM_STATE
from ...utils import _run_command

# Disable rpy2 warning
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda *args: None


def _get_levels(fasta_path):
    return list(
        set(
            linecache.getline(
                fasta_path,
                2,
            ).strip()
        )
    )


def make_distance_tree(dm, method="fastme"):
    """Make a tree from a distance matrix

    Parameters
    ----------
    dm : pd.DataFrame
        Distance matrix of size (n_classes, n_classes)
    method : str, optional
        Distance-based phylogenetic method, by default "fastme"
    """
    assert method in ("fastme", "nj", "upgma")

    if not isinstance(dm, pd.DataFrame):
        dm = pd.DataFrame(dm)

    with localconverter(ro.default_converter + pandas2ri.converter):
        importr("phangorn")

        ro.globalenv["dm"] = dm
        ro.globalenv["method"] = method

        newick = ro.r(
            """
            dm2 <- as.matrix(dm)

            if (method == "fastme") {
                tree <- fastme.bal(dm2)
            } else if (method == "upgma") {
                tree <- upgma(dm2)
            } else {
                tree <- bionj(dm2)
            }

            write.tree(tree)
            """
        )

    return newick[0]


def make_parsimony_tree(fasta_path, method="pratchet", **kwargs):
    # TODO: implement other parsimony methods?
    assert method in ("pratchet",)

    with localconverter(ro.default_converter):
        importr("phangorn")

        ro.globalenv["fasta_path"] = fasta_path
        # ro.globalenv["method"] = method
        ro.globalenv["lvls"] = ro.vectors.StrVector(_get_levels(fasta_path))

        newick = ro.r(
            """
            data <- read.phyDat(file = fasta_path, type = "USER", format = "fasta", levels = lvls)

            parsimony_tree <- pratchet(data, trace = 0)

            # add branch lengths
            parsimony_tree <- acctran(parsimony_tree, data)

            write.tree(parsimony_tree)
            """
        )

    with open(f"{fasta_path}.nwk", "w", encoding="utf-8") as f:
        f.write(newick[0])


def make_raxml_tree(
    fasta_path,
    raxml_ng_alias="raxml-ng",
    model="BIN+G+F",
    n_threads=DEFAULT_THREADS_TREE,
    **kwargs,
):
    command = (
        f"{raxml_ng_alias} --msa {fasta_path} --model {model} "
        f"--threads {n_threads} --seed {RANDOM_STATE} --nofiles interim"
    )

    _run_command(command)


def make_iqtree_tree(
    fasta_path,
    iqtree_alias="iqtree2",
    model="MFP",
    n_threads=DEFAULT_THREADS_TREE,
    bootstrap=False,
    **kwargs,
):
    command = f"{iqtree_alias} -s {fasta_path} -m {model} -T {n_threads} --seed {RANDOM_STATE}"

    if bootstrap:
        command += " -bb 1000 -redo"

    if "ORDERED" in model:
        command += " -safe"

    _run_command(command)
