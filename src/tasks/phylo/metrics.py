"""Metrics for comparing phylogenetic trees."""

import rpy2
import rpy2.robjects as ro
from ete3 import Tree
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# Disable rpy2 warning
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda *args: None


def robinson_foulds(ftry, ftrue):
    tree = Tree(ftry)
    ref_tree = Tree(ftrue)
    rf, rf_max, *_ = tree.robinson_foulds(ref_tree, unrooted_trees=True)
    return rf / rf_max


def generalized_robinson_foulds(ftry, ftrue):
    """
    Calculate the generalized Robinson-Foulds distance between two trees.
    """
    with localconverter(ro.default_converter + numpy2ri.converter):
        importr("ape")
        importr("TreeDist")

        ro.globalenv["newick_try"] = ftry
        ro.globalenv["newick_ref"] = ftrue

        rf_generalized = ro.r(
            """
            tree_try <- read.tree(newick_try)
            tree_ref <- read.tree(newick_ref)

            TreeDistance(tree_try, tree_ref)
            """
        )

    return rf_generalized[0]


def quartet_similarity(ftry, fref):
    with localconverter(ro.default_converter + numpy2ri.converter):
        importr("ape")
        importr("Quartet")

        ro.globalenv["ftry"] = ftry
        ro.globalenv["fref"] = fref

        s2r = ro.r(
            """
            tree_try <- read.tree(ftry)
            tree_ref <- read.tree(fref)

            # Make sure tips match
            s2r <- NA

            tip_diffs <- setdiff(tree_try$tip.label, tree_ref$tip.label)

            if (length(tip_diffs) > 0) {
                tree_try <- drop.tip(tree_try, tip_diffs)
            }

            if (length(setdiff(tree_try$tip.label, tree_ref$tip.label)) == 0) {
                status <- QuartetStatus(tree_try, cf = tree_ref)

                s2r <- SimilarityToReference(
                    status,
                    similarity = FALSE,
                    normalize = TRUE
                )
            }

            s2r
            """
        )

    return s2r[0]
