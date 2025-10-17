"""Download the FLEURS-R dataset"""

import json
import os
import subprocess
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from pathlib import Path

import requests
from tqdm import tqdm

DOWNLOAD_URL = "https://huggingface.co/datasets/google/fleurs-r/resolve/main/data"
FLEURS_R_LANGUAGES = "data/metadata/fleurs-r/languages.json"
TIMEOUT = 10


def parse_args():
    """Parse arguments for FLEURS-R data preparation"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default="data/datasets",
        help="Output dataset folder. Example: data/datasets",
    )
    return parser.parse_args()


def download(output_dir):
    """Download the FLEURS-R dataset into output_dir"""

    with open(FLEURS_R_LANGUAGES, "r", encoding="utf-8") as f:
        langs = list(json.load(f).keys())

    for i, lang in enumerate(langs):
        print(f"Downloading data from language {i+1}/{len(langs)}: {lang}")
        lang_folder = f"{output_dir}/fleurs-r/{lang}/{lang}"
        audio_folder = f"{lang_folder}/audio"
        os.makedirs(lang_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        for split in ["train", "dev", "test"]:
            # Donwload transcripts
            transcript_url = f"{DOWNLOAD_URL}/{lang}/{split}.tsv"
            response = requests.get(transcript_url, timeout=TIMEOUT)

            if response.status_code == 200:
                print(f"Downloading {transcript_url}")

                with open(
                    f"{lang_folder}/{os.path.basename(transcript_url)}", "wb"
                ) as ft:
                    ft.write(response.content)
            else:
                warnings.warn(f"File not found: {transcript_url}", UserWarning)
                continue

            # Download audio
            split_folder = f"{audio_folder}/{split}"
            os.makedirs(split_folder, exist_ok=True)

            audio_url = f"{DOWNLOAD_URL}/{lang}/audio/{split}.tar.gz"

            response = requests.get(audio_url, timeout=TIMEOUT)

            if response.status_code == 200:
                print(f"Downloading {audio_url}")
                with open(f"{split_folder}/{os.path.basename(audio_url)}", "wb") as fa:
                    fa.write(response.content)


def unzip(output_dir):
    """Decompress all the downloaded tar.gz files"""
    tar_files = sorted(glob(f"{output_dir}/fleurs-r/*/*/audio/*/*.tar.gz"))
    for tf in tqdm(tar_files, desc="Unzipping .tar.gz files"):
        tf_parent = str(Path(tf).parent)
        commands = (
            f"cd {tf_parent} && tar -xzf {os.path.basename(tf)} --strip-components=1"
        )
        subprocess.run(commands, check=True, shell=True)


def main():
    """Run main loop"""
    args = parse_args()

    download(output_dir=args.output_dir)

    unzip(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
