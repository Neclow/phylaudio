"""MLP-based language identification from audio embeddings.

Modules:
    classifier — MLP architecture with optional STE binarization, LightningMLP training wrapper.
    train      — End-to-end training/evaluation loop, wandb/CSV logging, CLI argument parsing.
"""

from .train import fit_predict, parse_lid_args

__all__ = ["fit_predict", "parse_lid_args"]
