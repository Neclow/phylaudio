"""Generate BEAST2 XML files from per-sentence tree alignments.

Inputs:  a per-sentence run UUID
Options: a concatenation mode, a selection criterion, and a fraction or number of sentences to keep
Flow:    Get per-sentence tree statistics
                    |
                    |
                    v
        Select top-p% by {brsupport,stemminess,clock}
                    |
                    |
                    v
                Merge/aggregate FASTAs
                    |
                    |
                    v
                Map taxon IDs
                    |
                    |
                    v fill XML templates
Outputs: input_vN.xml, prior_vN.xml, ns_vN/*.xml (n: version number)
"""

import json
import os
import sys
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
from src.data.nlp import load_sentence_filter
from src.tasks.phylo.fasta import merge_fastas, to_beast, vote_fastas

NS_TEMPLATE_DIR = f"{DEFAULT_BEAST_TEMPLATE_DIR}/ns"


def _next_version(beast_p_dir):
    """Find the next version number for input_vX.xml / prior_vX.xml.

    Scans `beast_p_dir` for existing input_v*.xml files and returns N+1.
    If the directory doesn't exist yet, returns 1.
    """
    if not os.path.isdir(beast_p_dir):
        return 1
    existing = glob(f"{beast_p_dir}/input_v*.xml")
    if not existing:
        return 1
    versions = []
    for f in existing:
        stem = Path(f).stem  # e.g. "input_v3"
        try:
            versions.append(int(stem.split("_v")[1]))
        except (IndexError, ValueError):
            continue
    return max(versions) + 1 if versions else 1


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
        "--mode",
        type=str,
        default="concat",
        choices=("concat", "vote"),
        help="concat: concatenate the top sentences ranked by --by; "
        "vote: per-site majority vote over all sentences (ignores -p/-n/--by)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-p",
        type=float,
        default=None,
        help="Fraction of sentences to keep (0-1, required for --mode concat unless -n is given)",
    )
    group.add_argument(
        "-n",
        type=int,
        default=None,
        help="Absolute number of sentences to keep (required for --mode concat unless -p is given)",
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
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        choices=(None, "ner"),
        help="Sentence filter: 'ner' removes sentences with proper nouns or named entities",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
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

    # Higher is better for brsupport/stemmy; lower is better for clock (CoV) / fscore (formality)
    ascending_by = {"brsupport": False, "stemmy": False, "clock": True, "fscore": True}

    if args.mode == "concat":
        if args.p is None and args.n is None:
            raise ValueError("-p or -n is required for --mode concat")
        if args.p is not None:
            assert 0 < args.p <= 1.0
        if args.n is not None:
            assert args.n > 0
        if args.by not in ascending_by:
            raise ValueError(
                f"Unknown sort criterion '{args.by}'. Choose from: {list(ascending_by.keys())}"
            )
        ascending = ascending_by[args.by]
    else:
        if args.p is not None or args.n is not None:
            warnings.warn("-p/-n is ignored with --mode vote", UserWarning)

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

    with open(f"{run_dir}/cfg.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        dataset = cfg["dataset"]

    # === Sentence selection funnel ===
    stats_file = (
        f"{run_dir}/_stats_{splits_label}.csv"
        if splits_label
        else f"{run_dir}/_stats.csv"
    )
    df = pd.read_csv(stats_file, index_col=0)
    print(f"\nLoaded {len(df)} sentences from {stats_file}")

    if splits_label:
        print(
            f"--{'include' if args.include else 'exclude'} {args.include or args.exclude}: {len(df)} sentences ({splits_label})"
        )

    if args.filter:
        allowed = load_sentence_filter(dataset, args.filter)
        df = df[df.index.map(lambda x: "_".join(x.split("_")[:2]) in allowed)]
        print(f"Using {args.filter} filter: {len(df)} sentences")

    if args.mode == "concat" and args.by == "fscore":
        spacy = pd.read_csv(
            f"{DEFAULT_METADATA_DIR}/{dataset}/spacy.csv", index_col="split_id"
        )
        df["fscore"] = df.index.map(
            lambda x: (
                spacy.loc["_".join(x.split("_")[:2]), "f_score"]
                if "_".join(x.split("_")[:2]) in spacy.index
                else float("nan")
            )
        )
        df = df.dropna(subset=["fscore"])

    if args.mode == "vote":
        sub_df = df
        print(f"Vote mode: using all {len(sub_df)} sentences")
    else:
        df_sorted = df.sort_values(by=args.by, ascending=ascending)
        if args.n is not None:
            n_keep = min(args.n, len(df_sorted))
            sub_df = df_sorted.iloc[:n_keep]
            print(f"Selecting top {args.n} by {args.by}: {len(sub_df)} sentences")
        else:
            n_keep = int(args.p * len(df_sorted))
            sub_df = df_sorted.iloc[:n_keep]
            print(f"Selecting top {args.p}% by {args.by}: {len(df_sorted)} sentences")

    input_files = [f"{run_dir}/{x}" for x in sub_df.index.to_list()]

    with open(
        f"{DEFAULT_METADATA_DIR}/{dataset}/languages.json", "r", encoding="utf-8"
    ) as f:
        languages = json.load(f)

    # Build output directory name
    if args.mode == "vote":
        beast_p_dir = f"{DEFAULT_BEAST_DIR}/{Path(args.run_id).stem}/vote"
    else:
        if args.n is not None:
            beast_p_dir = f"{DEFAULT_BEAST_DIR}/{Path(args.run_id).stem}/{args.n}"
        else:
            beast_p_dir = f"{DEFAULT_BEAST_DIR}/{Path(args.run_id).stem}/{args.p:.2f}"
        beast_p_dir += f"_{args.by}"
    if splits_label:
        beast_p_dir += f"_{splits_label}"
    if args.filter:
        beast_p_dir += f"_{args.filter}"
    os.makedirs(beast_p_dir, exist_ok=True)

    # Build FASTA alignment
    merged_file = f"{beast_p_dir}/{DEFAULT_MERGED_FASTA_FILE}"

    if args.mode == "vote":
        vote_fastas(input_files=input_files, output_file=merged_file)
    else:
        merge_fastas(
            input_files=input_files,
            output_file=merged_file,
            sequence_ids=list(languages.keys()),
        )

    # Map sequence IDs to reference names
    mapped_file = f"{beast_p_dir}/{DEFAULT_MAPPED_FASTA_FILE}"
    n_taxa = 0
    seq_len = 0

    with open(mapped_file, "w", encoding="utf-8") as f_out:
        for record in SeqIO.parse(merged_file, "fasta"):
            language = str(record.id)
            seq = str(record.seq)

            if language in languages and args.key in languages[language]:
                f_out.write(
                    f">{languages[language][args.key].replace(' ', '')}\n{seq}\n"
                )
                n_taxa += 1
                seq_len = len(seq)

    version = _next_version(beast_p_dir)
    xml_template = f"{DEFAULT_BEAST_TEMPLATE_DIR}/input_v1.xml"
    prior_template = f"{DEFAULT_BEAST_TEMPLATE_DIR}/prior_v1.xml"
    ns_templates = sorted(glob(f"{NS_TEMPLATE_DIR}/input_ns_*.xml"))

    print("\nWill generate:")
    print(f"\tVersion:    v{version}")
    print(f"\tOutput dir: {beast_p_dir}/")
    print(f"\tSentences:  {len(input_files)}")
    print(f"\tTaxa:       {n_taxa}")
    print(f"\tSites:      {seq_len}")
    files = [f"input_v{version}.xml"]
    if os.path.exists(prior_template):
        files.append(f"prior_v{version}.xml")
    if ns_templates:
        files.append(f"ns_v{version}/ ({len(ns_templates)} models)")
    print(f"\tFiles:      {', '.join(files)}")

    if not args.yes:
        confirm = input("\nProceed? [y/N] ")
        if confirm.lower() != "y":
            # Clean up FASTA files we already wrote
            for f in (merged_file, mapped_file):
                if os.path.exists(f):
                    os.remove(f)
            print("Aborted.")
            sys.exit(0)

    # Save config
    with open(f"{beast_p_dir}/cfg.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "mode": args.mode,
                "p": args.p if args.mode == "concat" and args.p else None,
                "n": args.n if args.mode == "concat" and args.n else None,
                "by": args.by if args.mode == "concat" else None,
                "splits": splits_label,
                "filter": args.filter,
                "key": args.key,
                "n_sentences": len(input_files),
            },
            f,
            indent=4,
        )

    # Generate XMLs
    taxonsets = (
        None if cfg["glottocode"] != "indo1319" else _FLEURS_TO_INDO1319_FAMILIES
    )

    output_file = f"{beast_p_dir}/input_v{version}.xml"
    print(f"Generating {output_file}...")
    to_beast(
        input_file=mapped_file,
        output_file=output_file,
        template_beast_file=xml_template,
        taxonsets=taxonsets,
    )

    if os.path.exists(prior_template):
        prior_output = f"{beast_p_dir}/prior_v{version}.xml"
        print(f"Generating {prior_output}...")
        to_beast(
            input_file=mapped_file,
            output_file=prior_output,
            template_beast_file=prior_template,
            taxonsets=taxonsets,
        )

    if ns_templates:
        ns_dir = f"{beast_p_dir}/ns_v{version}"
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
