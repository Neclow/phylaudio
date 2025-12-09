# pylint: disable=invalid-name

"""Summarise results from sentence tree inference runs."""

import json
import multiprocessing
import os
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from glob import glob

import ete3
import pandas as pd
import rpy2
from tqdm import tqdm

from src._config import DEFAULT_PER_SENTENCE_DIR
from src.models._model_zoo import MODEL_ZOO
from src.tasks.phylo.metrics import (
    generalized_robinson_foulds,
    quartet_similarity,
    robinson_foulds,
)

IGNORE_COLUMNS = ["dbs", "ebs", "device", "Commit"]
METRICS = {
    "rf_norm": robinson_foulds,
    "rf_generalized": generalized_robinson_foulds,
    "s2r": quartet_similarity,
}
OUTPUT_FILES = {"none": "_trees.txt", "astral4": "_trees_astral4.txt"}
REFERENCE_DIR = "data/trees/references/processed"


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "indir",
        help=f"Sentence tree directory in {DEFAULT_PER_SENTENCE_DIR}",
    )
    parser.add_argument(
        "--by",
        default="rf_generalized",
        choices=("rf_generalized", "rf_norm", "s2r"),
        help="Tree metric to sort the output by",
    )
    parser.add_argument(
        "-r",
        "--ref",
        default="iecor",
        choices=("iecor", "gled", "glottolog", "asjp"),
        type=str,
        help="Reference tree.",
    )
    parser.add_argument(
        "-nt",
        "--n-threads",
        default=4,
        type=int,
        help="Number of parallel threads to use.",
    )
    parser.add_argument(
        "-ot", "--output-type", default="astral4", choices=("astral4", "none"), type=str
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def extract_metrics_single(cfg_file, ref, output_tree_name):
    # Load config
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = pd.Series(json.load(f))

    # Check summary tree file exists
    tree_file = f"{os.path.dirname(cfg_file)}/{output_tree_name}"
    if not os.path.exists(tree_file):
        raise FileNotFoundError(f"Tree file not found: {tree_file}")

    # Reference tree file
    ref_tree_file = f"{REFERENCE_DIR}/{ref}.nwk"
    if not os.path.exists(ref_tree_file):
        raise FileNotFoundError(f"Reference tree file not found: {ref_tree_file}")

    # Augment cfg with model defaults if not specified
    model_cfg = MODEL_ZOO[cfg["model_id"]]
    cfg["max_length"] = cfg.get("max_length") or model_cfg["max_length"]
    cfg["dtype"] = cfg.get("dtype") or model_cfg["dtype"]

    for metric_name, metric_func in METRICS.items():
        key = f"{metric_name}_{ref}"
        try:
            cfg[key] = metric_func(tree_file, ref_tree_file)
        except (
            ete3.parser.newick.NewickError,
            rpy2.rinterface_lib.embedded.RRuntimeError,
            ZeroDivisionError,
        ) as err:
            warnings.warn(
                f"Error encountered for tree1={tree_file} and tree2={ref_tree_file}: {err}"
            )
            cfg[key] = float("nan")

    return cfg


if __name__ == "__main__":
    args = parse_args()

    # Parse cfg args of all runs (cfg.json)
    cfg_files = glob(f"{DEFAULT_PER_SENTENCE_DIR}/{args.indir}/*/cfg.json")
    n_files = len(cfg_files)
    assert n_files > 0, f"No runs found in {DEFAULT_PER_SENTENCE_DIR}/{args.indir}"
    print(f"Found {n_files} runs.")
    output_tree_name = OUTPUT_FILES[args.output_type]
    # Output file
    output_file = f"{DEFAULT_PER_SENTENCE_DIR}/{args.indir}/summary.csv"

    extract_fn = partial(
        extract_metrics_single,
        ref=args.ref,
        output_tree_name=output_tree_name,
    )
    with multiprocessing.Pool(processes=args.n_threads) as pool:
        work = pool.imap_unordered(extract_fn, cfg_files)
        outputs = list(tqdm(work, total=len(cfg_files), desc="Extracting metrics"))

    print("Gathering results...")

    by = f"{args.by}_{args.ref}"
    df = (
        pd.concat(outputs, axis=1)
        .T.set_index("run_id")
        .drop(columns=IGNORE_COLUMNS, errors="ignore")
        .infer_objects()
        .sort_values(by=by)
    )

    df.to_csv(output_file, float_format="%.4f")

    print("Done")
