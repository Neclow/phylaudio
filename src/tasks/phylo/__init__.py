"""Phylogenetic inference utilities.

Modules:
    beast      — BEAST run directory resolution, MCC tree discovery.
    fasta      — FASTA merging (concatenation, majority vote), BEAST XML injection/extraction.
    metrics    — Tree comparison: Robinson-Foulds (normalized, generalized), quartet similarity.
    newick     — Newick leaf-label renaming (FLEURS keys to reference names).
    nexus      — PhyloWriter classes for writing Nexus tree files from discrete/distance data.
    splitstree — Delta score extraction from SplitsTree .stree6 files.
    tree       — Tree building wrappers: IQ-TREE, FastME, RAxML-NG, phangorn pratchet.
"""
