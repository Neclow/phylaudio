import warnings
import xml
from pathlib import Path

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

from ..._config import MIN_LANGUAGES
from ...utils import _count_file_lines


def merge_fastas(input_files, sequence_ids, filler="?", output_file=None):
    """Merge several fastas into one supermatrix"""
    skipped = []

    seqs = {}

    for file in tqdm(input_files, desc="Merging FASTA files"):
        n_lines = _count_file_lines(file)
        if n_lines < 2 * MIN_LANGUAGES:
            skipped.append(
                (
                    Path(file).stem,
                    f"File has less than {MIN_LANGUAGES} sequences ({n_lines})",
                )
            )
            continue

        file_ids = []

        for record in SeqIO.parse(file, "fasta"):
            language = str(record.id)
            file_ids.append(language)

            seq = str(record.seq)

            if language in seqs:
                seqs[language] += seq
            else:
                seqs[language] = seq

        gap_sequence = filler * len(seq)

        for missed_language in set(sequence_ids) - set(file_ids):
            if missed_language in seqs:
                seqs[missed_language] += gap_sequence
            else:
                seqs[missed_language] = gap_sequence

    if len(skipped) > 0:
        warnings.warn(f"{len(skipped)} warnings encountered: {skipped}", UserWarning)

    print(
        f"After merge: {len(seqs)} sequences; alignment length: {len(next(iter(seqs.values())))}."
    )

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for language, seq in seqs.items():
                if set(seq) != set(filler):
                    f_out.write(f">{language}\n{seq}\n")
    else:
        return seqs


def vote_fastas(input_files, output_file=None):
    """Aggregate several fastas into one alignment by per-site majority vote.

    All input fastas must share the same alignment length. For each sequence id
    and site, the output character is the majority state ('0' or '1') across the
    files where that sequence is present and the site is not missing; exact ties
    and all-missing sites become '?'.
    """
    skipped = []

    ones = {}
    known = {}
    length = None

    for file in tqdm(input_files, desc="Voting over FASTA files"):
        n_lines = _count_file_lines(file)
        if n_lines < 2 * MIN_LANGUAGES:
            skipped.append(
                (
                    Path(file).stem,
                    f"File has less than {MIN_LANGUAGES} sequences ({n_lines})",
                )
            )
            continue

        for record in SeqIO.parse(file, "fasta"):
            language = str(record.id)
            seq = np.frombuffer(str(record.seq).encode(), dtype=np.uint8)

            if length is None:
                length = seq.size
            elif seq.size != length:
                raise ValueError(
                    f"Alignment length mismatch in {file}: {seq.size} != {length}"
                )

            if language not in ones:
                ones[language] = np.zeros(length, dtype=np.int64)
                known[language] = np.zeros(length, dtype=np.int64)

            is_one = seq == ord("1")
            ones[language] += is_one
            known[language] += is_one | (seq == ord("0"))

    if len(skipped) > 0:
        warnings.warn(f"{len(skipped)} warnings encountered: {skipped}", UserWarning)

    seqs = {}
    for language, n_ones in ones.items():
        n_known = known[language]
        chars = np.where(
            2 * n_ones > n_known, "1", np.where(2 * n_ones < n_known, "0", "?")
        )
        seqs[language] = "".join(chars)

    print(
        f"After vote: {len(seqs)} sequences; alignment length: {length}."
    )

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for language, seq in seqs.items():
                if set(seq) != {"?"}:
                    f_out.write(f">{language}\n{seq}\n")
    else:
        return seqs


def to_beast(input_file, output_file, template_beast_file, taxonsets=None):
    """Convert a FASTA file to a BEAST XML file.

    Parameters
    ----------
    input_file : Path-like object
        Input FASTA file
    output_file : Path-like object
        Output BEAST XML file
    template_beast_file : Path-like object
        Template BEAST XML file
    """
    sequences = {}
    for record in SeqIO.parse(input_file, "fasta"):
        taxon = record.id
        if taxon in sequences:
            warnings.warn(
                (
                    f"Duplicate sequence ID '{taxon}' found in FASTA file '{input_file}'. "
                    "Keeping the first occurrence."
                ),
                UserWarning,
            )
            continue
        sequences[taxon] = str(record.seq)

    print(f"Total sequences in FASTA: {len(sequences)}")

    tree = xml.etree.ElementTree.parse(template_beast_file)
    root = tree.getroot()

    # Prune template taxonset members that are absent from the alignment
    # (e.g. Afrikaans in datasets restricted to dev/test sentences)
    for taxonset_elm in root.iter("taxonset"):
        for taxon_elm in list(taxonset_elm.findall("taxon")):
            name = taxon_elm.get("id") or taxon_elm.get("idref")
            if name not in sequences:
                taxonset_elm.remove(taxon_elm)
                warnings.warn(
                    (
                        f"Pruned taxon '{name}' (absent from '{input_file}') "
                        f"from taxonset '{taxonset_elm.get('id')}'"
                    ),
                    UserWarning,
                )
        if len(taxonset_elm.findall("taxon")) == 1:
            warnings.warn(
                f"Taxonset '{taxonset_elm.get('id')}' has a single taxon after pruning",
                UserWarning,
            )

    data_elm = root.findall("data")[-1]

    # Clear existing sequence elements from the data section
    for sequence_elm in data_elm.findall("sequence"):
        data_elm.remove(sequence_elm)

    # Determine totalcount from the sequence data
    all_states = set()
    for seq in sequences.values():
        all_states.update(seq)
    all_states -= {"?", "-"}
    totalcount = str(max(int(c) for c in all_states) + 1)

    # Fill the data section directly from the FASTA sequences
    for taxon, seq in sequences.items():
        sequence_elm = xml.etree.ElementTree.SubElement(data_elm, "sequence")
        sequence_elm.set("id", f"seq_{taxon}")
        sequence_elm.set("spec", "Sequence")
        sequence_elm.set("taxon", taxon)
        sequence_elm.set("totalcount", totalcount)
        sequence_elm.set("value", seq)

    xml.etree.ElementTree.indent(tree, space="    ")

    # TODO: fill in taxonsets if provided

    if len(data_elm.findall("sequence")) >= MIN_LANGUAGES:
        tree.write(output_file)
        print(f"Written BEAST XML to {output_file}")
    else:
        print(f"{Path(input_file).stem} has less than {MIN_LANGUAGES} languages")


def from_beast(input_file, output_file):
    tree = xml.etree.ElementTree.parse(input_file)
    root = tree.getroot()
    data_elm = root.findall("data")[-1]

    with open(output_file, "w", encoding="utf-8") as f:
        for sequence_elm in data_elm.findall("sequence"):
            sequence_content = sequence_elm.attrib
            content = f">{sequence_content['taxon']}\n{sequence_content['value']}\n"
            f.write(content)


def to_numpy(fa_path):
    """
    Parse a binary FASTA alignment (0/1/? characters) into a numpy matrix.
    Missing values (?) are imputed with column means; all-missing columns get 0.0.

    Returns (X, labels) where X is (n_languages, n_sites) and labels is a list of names.
    """
    seqs = {}
    name = None
    with open(fa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                seqs[name] = ""
            else:
                seqs[name] += line

    labels = list(seqs.keys())
    n = len(labels)
    L = len(seqs[labels[0]])

    X_raw = np.full((n, L), np.nan)
    for i, nm in enumerate(labels):
        for j, c in enumerate(seqs[nm]):
            if c != "?":
                X_raw[i, j] = int(c)

    col_means = np.nanmean(X_raw, axis=0)
    # If an entire column is NaN, set its mean to 0.5
    col_means = np.where(np.isnan(col_means), 0.5, col_means)

    X = X_raw.copy()
    for j in range(L):
        missing = np.isnan(X[:, j])
        X[missing, j] = col_means[j]

    return X, labels
