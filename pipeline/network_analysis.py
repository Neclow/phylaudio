# 1. Export beast XML to FASTA
# 2. Run NeighborNet workflow on FASTA files

import os
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from Bio import SeqIO

from src._config import DEFAULT_MAPPED_FASTA_FILE, DEFAULT_SPLITSTREE_FASTA_FILE
from src.tasks.phylo.fasta import from_beast


def parse_args():
    parser = ArgumentParser(
        description="Run NeighborNet analysis on BEAST output FASTA files.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input BEAST XML file",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="src/tasks/phylo/splitstree_default.wflow6",
        help="Path to the SplitsTree6 workflow file (.wflow6).",
    )
    parser.add_argument(
        "--workflow_runner",
        type=str,
        default="extern/splitstree/tools/workflow-run",
        help="Path to the workflow runner script.",
    )
    parser.add_argument(
        "--char-map",
        type=str,
        default="?->-;",
        help="Character mapping for sequences (key1->value1;key2->value2;...;).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing output files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Convert BEAST XML to FASTA
    print("Converting BEAST XML to FASTA...")
    dirname = os.path.dirname(args.input)
    fasta_file = f"{dirname}/{DEFAULT_MAPPED_FASTA_FILE}"

    if not os.path.exists(fasta_file) or args.overwrite:
        from_beast(args.input, fasta_file)

    maps = args.char_map.split(";")
    char_map = {}
    for m in maps:
        if "->" in m:
            k, v = m.split("->")
            char_map[k] = v

    splitstree_input = f"{dirname}/{DEFAULT_SPLITSTREE_FASTA_FILE}"
    splitstree_output = os.path.splitext(splitstree_input)[0] + ".stree6"
    with open(splitstree_input, "w", encoding="utf-8") as f_out:
        records = SeqIO.parse(fasta_file, "fasta")
        for record in records:
            seq = str(record.seq)
            for k, v in char_map.items():
                seq = seq.replace(k, v)
            f_out.write(f">{record.id}\n{seq}\n")

    # Step 2: Run SplitsTree workflow
    print("Running SplitsTree workflow...")
    subprocess.run(
        [
            "bash",
            args.workflow_runner,
            "-i",
            splitstree_input,
            "-w",
            args.workflow,
            "-f",
            "FastA",
            "-o",
            splitstree_output,
        ],
        check=True,
    )

    print(
        (
            "By default, we use the LogDet distance transformation and NeighborNet in the workflow. "
            "Use the SplitsTree6 GUI to modify the analysis as needed."
            f"Or modify the workflow file at {args.workflow} to customize the analysis."
        )
    )
