"""Utilities for locating BEAST run directories and tree files."""

import os
from glob import glob

from src._config import DEFAULT_BEAST_DIR


def resolve_beast_dir(run_id: str, subdir: str) -> str:
    """Resolve run_id + subdir prefix to a BEAST run directory."""
    if os.path.isdir(run_id):
        beast_root = run_id
    else:
        matches = [m for m in glob(f"{DEFAULT_BEAST_DIR}/{run_id}*") if os.path.isdir(m)]
        if not matches:
            raise FileNotFoundError(
                f"No BEAST run matching '{run_id}' in {DEFAULT_BEAST_DIR}/"
            )
        if len(matches) > 1:
            raise ValueError(f"Ambiguous run_id '{run_id}': {matches}")
        beast_root = matches[0]

    subdir_matches = [
        m for m in glob(f"{beast_root}/{subdir}*") if os.path.isdir(m)
    ]
    if not subdir_matches:
        raise FileNotFoundError(
            f"No subdirectory matching '{subdir}' in {beast_root}/"
        )
    if len(subdir_matches) > 1:
        raise ValueError(f"Ambiguous subdir '{subdir}': {subdir_matches}")
    return subdir_matches[0]


def find_beast_mcc(beast_dir: str, version: int | None = None) -> str:
    """Find the tree file in a BEAST directory.

    If version is given, returns input_v{version}.mcc directly.
    Otherwise globs for *.mcc then *.nex.
    """
    if version is not None:
        path = f"{beast_dir}/input_v{version}.mcc"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tree file not found: {path}")
        return path

    for pattern in ("*.mcc", "*.nex"):
        hits = glob(f"{beast_dir}/{pattern}")
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"Multiple {pattern} files in {beast_dir}: {hits}")
    raise FileNotFoundError(f"No .mcc or .nex tree file in {beast_dir}")
