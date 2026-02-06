# pylint: disable=redefined-outer-name
"""Download and process reference phylogenetic trees."""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import requests

from src._config import DEFAULT_BEAST_DIR, DEFAULT_METADATA_DIR, REFERENCE_TREES
from src.data.glottolog import filter_languages_from_glottocode


def parse_args():
    parser = ArgumentParser(
        description="Arguments for Glottolog lineage extraction",
        formatter_class=ArgumentDefaultsHelpFormatter,
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
        "-g",
        "--glottolog-dir",
        default="extern/glottolog",
        help="Path to glottolog folder",
    )
    parser.add_argument(
        "--glottocode",
        type=str,
        default="indo1319",
        help="Glottolog languoid code for the root of the tree to extract",
    )
    parser.add_argument(
        "--min-speakers",
        type=float,
        default=0.0,
        help="Minimum number of speakers for a language to be included (in millions)",
    )
    parser.add_argument(
        "--preserve-branch-length",
        action="store_true",
        help="Whether to preserve branch lengths when pruning trees",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing downloaded files",
    )
    return parser.parse_args()


# pylint: disable=line-too-long

EXTRA_IECOR_TREES = {
    "posterior": {
        "url": "https://share.eva.mpg.de/public.php/dav/files/E4Am2bbBA3qLngC/01_Main_Analysis_M3/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined.trees",
        "file": f"{DEFAULT_BEAST_DIR}/iecor/raw.trees",
    },
    "prior": {
        "url": "https://share.eva.mpg.de/public.php/dav/files/E4Am2bbBA3qLngC/01_Main_Analysis_M3/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined_PRIOR.trees",
        "file": f"{DEFAULT_BEAST_DIR}/iecor/prior/raw.trees",
    },
}
# pylint: enable=line-too-long


if __name__ == "__main__":
    args = parse_args()

    # Get json language data
    languages_filtered = filter_languages_from_glottocode(
        dataset=args.dataset,
        glottocode=args.glottocode,
        min_speakers=args.min_speakers,
    )

    for key, processor_utils in REFERENCE_TREES.items():
        # Prepare tree processor
        ProcessorCls = processor_utils["downloader"]
        extra_args = {}
        if key == "glottolog":
            extra_args["glottolog_dir"] = args.glottolog_dir
            extra_args["glottocode"] = args.glottocode
        elif key.startswith("asjp"):
            extra_args["asjp_url"] = processor_utils["asjp_url"]
        tree_processor = ProcessorCls(
            name=key,
            url=processor_utils.get("url", None),
            ext=processor_utils["ext"],
            **extra_args,
        )

        # Download raw tree file
        tree_processor.maybe_download(overwrite=args.overwrite)

        # Process tree file to only include relevant languages
        print(f"Processing: {key}...")
        tree_processor.process(
            languages_to_prune=languages_filtered,
            process_args=vars(args),
            preserve_branch_length=args.preserve_branch_length,
        )

    for key, extra_iecor_info in EXTRA_IECOR_TREES.items():
        print(f"Downloading extra IECoR trees: {key}...")
        if os.path.exists(extra_iecor_info["file"]) and not args.overwrite:
            print(
                f"Target file {extra_iecor_info['file']} already exists. Skipping download."
            )
        else:
            response = requests.get(
                extra_iecor_info["url"], allow_redirects=True, timeout=10
            )
            response.raise_for_status()
            with open(extra_iecor_info["file"], "wb") as f:
                f.write(response.content)

    print("Downloaded reference trees.")
