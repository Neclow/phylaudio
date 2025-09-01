# pylint: disable=invalid-name

import json
import os
import sys
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from pathlib import Path

import ete3
import pandas as pd
import rpy2

from src.models._model_zoo import MODEL_ZOO
from src.tasks.phylo.metrics import (
    generalized_robinson_foulds,
    quartet_similarity,
    robinson_foulds,
)

# Disable pandas warning
pd.set_option("future.no_silent_downcasting", True)

IGNORE_COLUMNS = ["dbs", "ebs", "device", "Commit"]
METRICS = {
    "rf_norm": robinson_foulds,
    "rf_generalized": generalized_robinson_foulds,
    "s2r": quartet_similarity,
}
OUTPUT_FILES = {"none": "_trees.txt", "astral4": "_trees_astral4.txt"}


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        help="Folder with cfg.json files",
        choices=(
            "pdist",
            "pdist_agg",
            "pdist_quantized",
            "pdist_test",
            "discrete",
            "discrete_ica128",
        ),
    )
    parser.add_argument("--dataset", default="fleurs", help="Dataset")
    parser.add_argument(
        "--glottocode",
        default="indo1319",
        help="Glottocode of a language family",
    )
    parser.add_argument(
        "--by",
        default="rf_generalized",
        choices=("rf_generalized", "rf_norm", "s2r"),
        help="Tree metric to sort the output by",
    )
    parser.add_argument(
        "--ref",
        default="heggarty2023",
        type=str,
        help="Reference tree to compare against. "
        "If not specified, the script will use the first available reference tree.",
    )
    parser.add_argument(
        "--output-tree", default="astral4", choices=("astral4", "none"), type=str
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_files = glob(f"data/{args.folder}/*/cfg.json")

    ref_tree_files = sorted(
        glob(f"data/_references/trees/processed/*{args.glottocode}_{args.dataset}.nwk")
    )

    assert len(ref_tree_files) > 0

    output_file = f"data/{args.folder}/summary_{args.glottocode}.csv"

    if os.path.exists(output_file) and not args.overwrite:
        df = pd.read_csv(output_file, index_col="run_id")
        idxs = df.index
    else:
        df = None
        idxs = []

    dfs = []

    for cf in cfg_files:
        with open(cf, "r", encoding="utf-8") as f:
            cfg = pd.Series(json.load(f))

        # fmt: off
        # to be skipped:
        # either the run is already saved and we don't overwrite
        # or the glottocode does not match
        if (
            (cfg["run_id"] in idxs and not args.overwrite) or
            cfg["glottocode"] != args.glottocode
        ):
            continue
        # fmt: on

        cfg_tree_file = f"{os.path.dirname(cf)}/{OUTPUT_FILES[args.output_tree]}"

        # cfg["timestamp"] = os.path.getmtime(cfg_tree_file)

        model_ids = cfg["model_id"].split("+")

        # Infer dtype and max length from model_ids
        default_max_length = ""
        default_dtype = ""

        for m in model_ids:
            default_max_length += f"{MODEL_ZOO[m]['max_length']}+"
            default_dtype += f"{MODEL_ZOO[m]['dtype']}+"

        cfg["max_length"] = cfg.get("max_length") or default_max_length[:-1]
        cfg["dtype"] = cfg.get("dtype") or default_dtype[:-1]

        for ref_tree_file in ref_tree_files:
            ref_stem = Path(ref_tree_file).stem.split("_")[0]
            # cfg[ref_stem] = {}

            for metric_name, metric_func in METRICS.items():
                try:
                    cfg[f"{ref_stem}/{metric_name}"] = metric_func(
                        cfg_tree_file, ref_tree_file
                    )
                except (
                    ete3.parser.newick.NewickError,
                    rpy2.rinterface_lib.embedded.RRuntimeError,
                    ZeroDivisionError,
                ) as err:
                    warnings.warn(
                        f"Error encountered for {cfg_tree_file} with {ref_tree_file}: {err}"
                    )
                    cfg[f"{ref_stem}/{metric_name}"] = float("nan")

        print(f"Adding new data: {cfg['run_id']}")
        dfs.append(cfg)

    if len(dfs) == 0:
        by = f"{args.by}_{Path(ref_tree_files[0]).stem}"
        print("No new data to aggregate")
        sys.exit(0)
    new_df = pd.concat(dfs, axis=1).T.set_index("run_id")

    # new_df["timestamp"] = pd.to_datetime(new_df.timestamp, unit="s")

    if args.folder.startswith == "discrete":
        default_values = {
            "q": 2,
            "method": "iqtree2",
            "iqtree_model": "MFP",
            "ckpt": "",
            "min_speakers": 0.0,
            "pca": float("nan"),
        }
    else:
        default_values = {
            "ckpt": "",
            "min_speakers": 0.0,
            "decomposition": "",
            "nc": float("nan"),
        }

    new_df.fillna(default_values, inplace=True)

    if df is not None:
        new_df = pd.concat([df, new_df.loc[:, df.columns]], axis=0)

    # drop constant-valued columns
    if new_df.shape[0] > 1:
        new_df = new_df.loc[:, new_df.nunique().values > 1]

    by = f"{args.ref}/{args.by}"
    new_df = (
        new_df.sort_values(by=by)
        .infer_objects()
        .drop(columns=IGNORE_COLUMNS, errors="ignore")
    )

    new_df.to_csv(output_file, float_format="%.4f")
