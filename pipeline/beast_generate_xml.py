import json
import os
import warnings
from argparse import ArgumentParser, MetavarTypeHelpFormatter
from glob import glob
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from src._config import (
    DEFAULT_BEAST_DIR,
    DEFAULT_MAPPED_FASTA_FILE,
    DEFAULT_MERGED_FASTA_FILE,
    DEFAULT_METADATA_DIR,
    DEFAULT_PER_SENTENCE_DIR,
)
from src.tasks.phylo.fasta import merge_fastas, to_beast

XML_TEMPLATE_FILE = f"{DEFAULT_BEAST_DIR}/template.xml"


def parse_args():
    """Parse command line arguments for generating BEAST XML files."""
    parser = ArgumentParser(
        description="Fill XML files with sequences from the best trees",
        formatter_class=MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "dataset",
        type=str,
        help=(
            "Dataset. Example: `fleurs`. "
            f"Has to have a folder in `{DEFAULT_METADATA_DIR}` with a `languages.json` file."
        ),
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=str,
        required=True,
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
        "--key",
        type=str,
        default="iecor",
        help="Reference field in language metadata to use for taxon names, by default 'iecor'",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum number of speakers per language",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assert 0 < args.p <= 1.0

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
    df = pd.read_csv(f"{run_dir}/_stats.csv", index_col=0)
    input_files = [
        f"{run_dir}/{x}" for x in df.iloc[: int(args.p * df.shape[0])].index.to_list()
    ]

    with open(
        f"{DEFAULT_METADATA_DIR}/{args.dataset}/languages.json", "r", encoding="utf-8"
    ) as f:
        languages = json.load(f)

    beast_p_dir = f"{DEFAULT_BEAST_DIR}/{Path(args.run_id).stem}/{args.p:.2f}"
    os.makedirs(beast_p_dir, exist_ok=True)

    merged_file = f"{beast_p_dir}/{DEFAULT_MERGED_FASTA_FILE}"

    merge_fastas(
        input_files=input_files,
        output_file=merged_file,
        sequence_ids=list(languages.keys()),
    )

    mapped_file = f"{beast_p_dir}/{DEFAULT_MAPPED_FASTA_FILE}"

    with open(mapped_file, "w", encoding="utf-8") as f_out:
        for record in SeqIO.parse(merged_file, "fasta"):
            language = str(record.id)
            seq = str(record.seq)

            if (
                language in languages
                and args.key in languages[language]
                and languages[language]["speakers"] >= args.min_speakers
            ):
                f_out.write(f">{languages[language][args.key]}\n{seq}\n")

    output_file = f"{beast_p_dir}/input.xml"
    print(f"Generating BEAST XML file ({output_file})...")
    to_beast(
        input_file=mapped_file,
        output_file=output_file,
        template_beast_file=XML_TEMPLATE_FILE,
    )

    print("Done")


if __name__ == "__main__":
    main()
