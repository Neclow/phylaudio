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

from src._config import DEFAULT_AUDIO_DIR, DEFAULT_METADATA_DIR

TIMEOUT = 10


def parse_args():
    """Parse arguments for FLEURS(-R) data downloading"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="fleurs-r",
        help="Dataset to download. Currently only `fleurs-r` is supported.",
    )

    return parser.parse_args()


def download(dataset):
    """Download the FLEURS(-R) dataset"""

    download_url = f"https://huggingface.co/datasets/google/{dataset}/resolve/main/data"

    metadata = f"{DEFAULT_METADATA_DIR}/{dataset}/languages.json"

    if not os.path.isfile(metadata):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata}. Please run `python pipeline/download_fleurs_metadata.py` first."
        )

    with open(metadata, "r", encoding="utf-8") as f:
        langs = list(json.load(f).keys())

    for i, lang in enumerate(langs):
        print(f"Downloading data from language {i+1}/{len(langs)}: {lang}")
        lang_folder = f"{DEFAULT_AUDIO_DIR}/{dataset}/{lang}/{lang}"
        audio_folder = f"{lang_folder}/audio"
        os.makedirs(lang_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        for split in ["train", "dev", "test"]:
            # Donwload transcripts
            transcript_url = f"{download_url}/{lang}/{split}.tsv"
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

            audio_url = f"{download_url}/{lang}/audio/{split}.tar.gz"

            response = requests.get(audio_url, timeout=TIMEOUT)

            if response.status_code == 200:
                print(f"Downloading {audio_url}")
                with open(f"{split_folder}/{os.path.basename(audio_url)}", "wb") as fa:
                    fa.write(response.content)


def unzip(dataset):
    """Decompress all the downloaded tar.gz files"""
    tar_files = sorted(glob(f"{DEFAULT_AUDIO_DIR}/{dataset}/*/*/audio/*/*.tar.gz"))
    for tf in tqdm(tar_files, desc="Unzipping .tar.gz files"):
        tf_parent = str(Path(tf).parent)
        commands = (
            f"cd {tf_parent} && tar -xzf {os.path.basename(tf)} --strip-components=1"
        )
        subprocess.run(commands, check=True, shell=True)


def main():
    """Run main loop"""
    args = parse_args()

    download(dataset=args.dataset)

    unzip(dataset=args.dataset)


if __name__ == "__main__":
    main()
