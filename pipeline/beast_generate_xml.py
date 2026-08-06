import json
import os
import warnings
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    MetavarTypeHelpFormatter,
)
from glob import glob
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from src._config import (
    _FLEURS_TO_INDO1319_FAMILIES,
    DEFAULT_BEAST_DIR,
    DEFAULT_BEAST_TEMPLATE_DIR,
    DEFAULT_MAPPED_FASTA_FILE,
    DEFAULT_MERGED_FASTA_FILE,
    DEFAULT_METADATA_DIR,
    DEFAULT_METADATA_KEY,
    DEFAULT_PER_SENTENCE_DIR,
)
from src.tasks.phylo.fasta import merge_fastas, to_beast

XML_TEMPLATE_FILE = f"{DEFAULT_BEAST_TEMPLATE_DIR}/input_v1.xml"
PRIOR_TEMPLATE_FILE = f"{DEFAULT_BEAST_TEMPLATE_DIR}/prior_v1.xml"
NS_TEMPLATE_DIR = f"{DEFAULT_BEAST_TEMPLATE_DIR}/ns"


class MixedFormatter(ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter):
    pass


def parse_args():
    """Parse command line arguments for generating BEAST XML files."""
    parser = ArgumentParser(
        description="Fill XML files with sequences from the best trees",
        formatter_class=MixedFormatter,
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run ID (or path to run directory)",
    )
    parser.add_argument(
        "-p",
        "--p",
        type=float,
        required=True,
        help="%% of sequences to keep",
    )
    parser.add_argument(
        "--by",
        type=str,
        default="brsupport",
        help="Criterion to select trees",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=DEFAULT_METADATA_KEY,
        help="Reference field in language metadata to use for taxon names",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="Comma-separated splits to include (e.g. dev,test)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated splits to exclude (e.g. train)",
    )
    return parser.parse_args()


VALID_SPLITS = ("train", "dev", "test")


def resolve_splits(include, exclude):
    """Resolve --include/--exclude to a sorted splits label (or None for all)."""
    if include and exclude:
        raise ValueError("--include and --exclude are mutually exclusive")
    if include:
        splits = sorted(include.split(","))
    elif exclude:
        excluded = set(exclude.split(","))
        splits = sorted(s for s in VALID_SPLITS if s not in excluded)
    else:
        return None
    for s in splits:
        if s not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{s}' (must be one of {VALID_SPLITS})")
    return "_".join(splits)


def main():
    args = parse_args()

    assert 0 < args.p <= 1.0

    # Higher is better for brsupport/stemmy; lower is better for clock (CoV)
    ascending_by = {"brsupport": False, "stemmy": False, "clock": True}
    if args.by not in ascending_by:
        raise ValueError(
            f"Unknown sort criterion '{args.by}'. Choose from: {list(ascending_by.keys())}"
        )
    ascending = ascending_by[args.by]

    splits_label = resolve_splits(args.include, args.exclude)

    if not os.path.isdir(args.run_id):
        potential_run_dirs = glob(f"{DEFAULT_PER_SENTENCE_DIR}/*/{args.run_id}")

        if len(potential_run_dirs) == 0:
            raise FileNotFoundError(
                f"No run directory found for run ID '{args.run_id}' in '{DEFAULT_PER_SENTENCE_DIR}'"
            )
        if len(potential_run_dirs) > 1:
            warnings.warn(
                (
                    f"Multiple run directories found for run ID '{args.run_id}': {potential_run_dirs} "
                    "Keeping the first one."
                ),
                UserWarning,
            )
        run_dir = potential_run_dirs[0]
    else:
        run_dir = args.run_id

    print(f"Using run directory: {run_dir}")
    stats_file = f"{run_dir}/_stats_{splits_label}.csv" if splits_label else f"{run_dir}/_stats.csv"
    df = pd.read_csv(stats_file, index_col=0)
    sub_df = df.sort_values(by=args.by, ascending=ascending).iloc[
        : int(args.p * df.shape[0])
    ]
    print(sub_df.describe())
    input_files = [f"{run_dir}/{x}" for x in sub_df.index.to_list()]

    with open(f"{run_dir}/cfg.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        dataset = cfg["dataset"]

    with open(
        f"{DEFAULT_METADATA_DIR}/{dataset}/languages.json", "r", encoding="utf-8"
    ) as f:
        languages = json.load(f)

    beast_p_dir = f"{DEFAULT_BEAST_DIR}/{Path(args.run_id).stem}/{args.p:.2f}"
    beast_p_dir += f"_{args.by}"
    if splits_label:
        beast_p_dir += f"_{splits_label}"
    os.makedirs(beast_p_dir, exist_ok=True)

    # Merge FASTA files
    merged_file = f"{beast_p_dir}/{DEFAULT_MERGED_FASTA_FILE}"

    merge_fastas(
        input_files=input_files,
        output_file=merged_file,
        sequence_ids=list(languages.keys()),
    )

    # Map sequence IDs to reference names in mapped FASTA file
    mapped_file = f"{beast_p_dir}/{DEFAULT_MAPPED_FASTA_FILE}"

    with open(mapped_file, "w", encoding="utf-8") as f_out:
        for record in SeqIO.parse(merged_file, "fasta"):
            language = str(record.id)
            seq = str(record.seq)

            if language in languages and args.key in languages[language]:
                f_out.write(
                    f">{languages[language][args.key].replace(' ', '')}\n{seq}\n"
                )

    output_file = f"{beast_p_dir}/input_v1.xml"
    print(f"Generating BEAST XML file ({output_file})...")

    taxonsets = (
        None if cfg["glottocode"] != "indo1319" else _FLEURS_TO_INDO1319_FAMILIES
    )
    to_beast(
        input_file=mapped_file,
        output_file=output_file,
        template_beast_file=XML_TEMPLATE_FILE,
        taxonsets=taxonsets,
    )

    if os.path.exists(PRIOR_TEMPLATE_FILE):
        prior_output = f"{beast_p_dir}/prior_v1.xml"
        print(f"Generating prior XML: {prior_output}")
        to_beast(
            input_file=mapped_file,
            output_file=prior_output,
            template_beast_file=PRIOR_TEMPLATE_FILE,
            taxonsets=taxonsets,
        )

    ns_templates = sorted(glob(f"{NS_TEMPLATE_DIR}/input_ns_*.xml"))
    if ns_templates:
        ns_dir = f"{beast_p_dir}/ns"
        os.makedirs(ns_dir, exist_ok=True)
        print(f"Generating {len(ns_templates)} NS XML files in {ns_dir}...")
        for template in ns_templates:
            ns_output = f"{ns_dir}/{Path(template).name}"
            to_beast(
                input_file=mapped_file,
                output_file=ns_output,
                template_beast_file=template,
                taxonsets=taxonsets,
            )

    print("Done")


if __name__ == "__main__":
    main()
