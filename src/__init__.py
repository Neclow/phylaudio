"""Phylaudio source library.

The `src` directory contains core, non-user-facing functions for the Phylaudio project.

Subpackages:
    models/   — Feature extractors (BaseFeatureExtractor interface, MODEL_ZOO registry).
                Backends: transformers (XLS-R, MMS, HuBERT), whisper, speechbrain
                (ECAPA-TDNN), nemo (AmberNet), opensmile, baseline CNNs.
    data/     — Dataset classes (AudioDataset, FleursParallelDataset), Glottolog
                language filtering, speaker population data, audio transforms.
    tasks/    — Pipeline logic: feature extraction (embedding, discretization,
                decomposition), language identification (MLP classifier, STE),
                phylogenetic inference (tree building, FASTA/Nexus I/O, BEAST
                utilities, metrics, SplitsTree integration).
"""
