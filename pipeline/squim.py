# pylint: disable=invalid-name

"""Non-intrusive speech quality estimation using torchaudio SQUIM"""

import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from src._config import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_METADATA_DIR,
    DEFAULT_ROOT_DIR,
    SAMPLE_RATE,
)


def parse_args():
    """Parse arguments for SQUIM audio quality estimation"""

    parser = ArgumentParser(
        description="Estimate audio quality metrics (PESQ, STOI, SI-SDR) using SQUIM",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="fleurs-r",
        type=str,
        help="Dataset name",
    )
    parser.add_argument(
        "--subset",
        default=None,
        type=str,
        choices=["train", "dev", "test"],
        help="Data subset (default: all splits)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device on which a torch.Tensor is or will be allocated",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Number of waveforms to process at once",
    )

    return parser.parse_args()


def load_audio(path, target_sr=SAMPLE_RATE):
    """Load and resample a single audio file to mono at target_sr"""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def main():
    """Main loop"""

    args = parse_args()

    # Load language metadata
    metadata_path = f"{DEFAULT_METADATA_DIR}/{args.dataset}/languages.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        languages = json.load(f)

    # Load SQUIM objective model
    print("(squim) Loading SQUIM objective model...")
    objective_model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(args.device)
    objective_model.eval()

    # Output directory
    output_dir = f"{DEFAULT_METADATA_DIR}/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    subsets = [args.subset] if args.subset else ["train", "dev", "test"]
    all_records = []

    with torch.no_grad(), torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16):
        for lang in sorted(languages.keys()):
            for subset in subsets:
                audio_dir = (
                    f"{DEFAULT_AUDIO_DIR}/{args.dataset}/{lang}/{lang}/audio/{subset}"
                )
                audio_files = sorted(glob(f"{audio_dir}/*.wav"))

                if not audio_files:
                    continue

                # Process in batches
                for batch_start in tqdm(
                    range(0, len(audio_files), args.batch_size),
                    total=(len(audio_files) + args.batch_size - 1) // args.batch_size,
                    desc=f"(squim) Processing {lang}/{subset}",
                ):
                    batch_files = audio_files[
                        batch_start : batch_start + args.batch_size
                    ]

                    # Load audio in parallel threads (I/O-bound)
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        waveforms = list(
                            pool.map(
                                lambda p: load_audio(p, target_sr=SAMPLE_RATE).squeeze(0),
                                batch_files,
                            )
                        )

                    # Pad to max length in batch
                    lengths = [w.shape[-1] for w in waveforms]
                    max_len = max(lengths)
                    padded = torch.stack(
                        [
                            torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
                            for w in waveforms
                        ]
                    ).to(args.device)

                    stoi, pesq, si_sdr = objective_model(padded)

                    for i, fpath in enumerate(batch_files):
                        all_records.append(
                            {
                                "language": lang,
                                "subset": subset,
                                "file": os.path.basename(fpath),
                                "stoi": stoi[i].item(),
                                "pesq": pesq[i].item(),
                                "si_sdr": si_sdr[i].item(),
                            }
                        )

    # Save results
    df = pd.DataFrame(all_records)
    output_file = f"{output_dir}/squim.csv"
    df.to_csv(output_file, index=False)
    print(f"(squim) Saved {len(df)} records to {output_file}")

    # Print per-language summary
    summary = df.groupby("language")[["stoi", "pesq", "si_sdr"]].mean()
    print("\n(squim) Per-language summary (mean):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
