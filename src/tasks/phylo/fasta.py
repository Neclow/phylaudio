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


def to_beast(input_file, output_file, template_beast_file, languages, ref="iecor"):
    skipped = []

    # Gather all sequences where the language is in IECOR
    sequences = {
        languages[sequence.id].get(ref, None): str(sequence.seq)
        for sequence in SeqIO.parse(input_file, "fasta")
    }

    del sequences[None]

    tree = xml.etree.ElementTree.parse(template_beast_file)
    root = tree.getroot()
    data_elm = root.findall("data")[-1]
    taxon_elm = root.findall("run")[0].findall(".//taxon")

    # Update the xml data with the matched fasta sequences
    for sequence_elm in data_elm.findall("sequence"):
        sequence_content = sequence_elm.attrib

        if sequence_content["taxon"] in sequences:
            sequence_content["value"] = sequences[sequence_content["taxon"]]
        else:
            data_elm.remove(sequence_elm)

    # Update the xml data with the matched taxon names
    for taxon_elm in root.findall("run")[0].findall(".//taxon"):
        taxon_content = taxon_elm.attrib

        if taxon_content["id"] not in sequences:
            taxonset_elm = root.findall("run")[0].findall(
                f".//taxon[@id='{taxon_content['id']}']..."
            )[0]
            taxonset_elm.remove(taxon_elm)

    if len(data_elm.findall("sequence")) >= MIN_LANGUAGES:
        tree.write(output_file)
    else:
        skipped.append(
            f"{Path(input_file).stem} has less than {MIN_LANGUAGES} languages"
        )


def zip_fastas(run_id):
    """Zip all fasta files in a directory"""
    command = (
        f"tar -czf data/discrete/{run_id}/_fastas.tar.gz data/discrete/{run_id}/*.fa"
    )

    _run_command(command)
