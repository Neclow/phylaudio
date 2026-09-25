#!/usr/bin/env bash
# Download, verify and unpack the Zenodo deposit (version 2) into the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="phylaudio_zenodo_v2.tar.gz"
URL="https://zenodo.org/api/records/22228727/files/${FILE}/content"
MD5="b82302681ea18378c854f21800ba17fb"

cd "$ROOT"

md5_of() {
    if command -v md5sum >/dev/null; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}

if [ -f "$FILE" ] && [ "$(md5_of "$FILE")" = "$MD5" ]; then
    echo "$FILE already downloaded and verified."
else
    echo "Downloading $FILE (~28 GB) from Zenodo..."
    curl -L --fail -C - -o "$FILE" "$URL"
    if [ "$(md5_of "$FILE")" != "$MD5" ]; then
        echo "Checksum mismatch for $FILE; delete it and re-run." >&2
        exit 1
    fi
fi

echo "Unpacking $FILE..."
# Skip input_v2b* (plain-MCMC test run in the v2 tarball, not used in the paper)
tar -xzf "$FILE" --exclude='*input_v2b*'
echo "Done."
