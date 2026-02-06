import warnings
import xml
from pathlib import Path

from Bio import SeqIO
from tqdm import tqdm

from ..._config import MIN_LANGUAGES
from ...utils import _count_file_lines, _run_command


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


def zip_fastas(run_id):
    """Zip all fasta files in a directory"""
    command = (
        f"tar -czf data/discrete/{run_id}/_fastas.tar.gz data/discrete/{run_id}/*.fa"
    )

    _run_command(command)
