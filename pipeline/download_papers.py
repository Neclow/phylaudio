"""Download the reference papers whose tables the supplementary notebook parses.

Inputs:  none
Options: none
Flow:    For each paper: skip if a verified copy exists
                    |
                    v
         Download from arXiv / the ACL Anthology, then check its MD5
Outputs: data/resources/{2111.09296,mls,vl107,voxpopuli}.pdf

The PDFs are not redistributed in the repository (copyright); they are fetched
at the exact versions the table-parsing code in plots/supplementary_v2.ipynb
was written against.
"""

import hashlib
import os

import requests

from src._config import DEFAULT_RESOURCES_DIR

TIMEOUT = 30

# filename, source URL, expected MD5
PAPERS = [
    ("2111.09296.pdf", "https://arxiv.org/pdf/2111.09296v3", "f61e4d38dddd888e8ea8c910b826a39b"),
    ("mls.pdf", "https://arxiv.org/pdf/2012.03411v2", "3caffc688ceec12bf768ad0deb220f9c"),
    ("vl107.pdf", "https://arxiv.org/pdf/2011.12998v1", "6aa131fcc1a1b8bf7783413071a0fc3d"),
    ("voxpopuli.pdf", "https://aclanthology.org/2021.acl-long.80.pdf", "ac673669cc2f9a858703cd480c96f763"),
]


def md5_of(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    os.makedirs(DEFAULT_RESOURCES_DIR, exist_ok=True)

    for filename, url, md5 in PAPERS:
        path = f"{DEFAULT_RESOURCES_DIR}/{filename}"
        if os.path.exists(path) and md5_of(path) == md5:
            print(f"{filename} already downloaded and verified.")
            continue

        print(f"Downloading {filename} from {url}")
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)

        if md5_of(path) != md5:
            raise ValueError(f"Checksum mismatch for {filename}; delete it and re-run.")


if __name__ == "__main__":
    main()
