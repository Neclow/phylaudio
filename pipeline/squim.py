# pylint: disable=invalid-name

"""Non-intrusive speech quality estimation using torchaudio SQUIM"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import pandas as pd
import torch
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE
from tqdm import tqdm

from src._config import DEFAULT_METADATA_DIR, DEFAULT_ROOT_DIR, SAMPLE_RATE
from src.data.datasets import FleursParallelDataset


def parse_args():
    """Parse arguments for SQUIM audio quality estimation"""

    parser = ArgumentParser(
        description="Estimate audio quality metrics (PESQ, STOI, SI-SDR) using SQUIM",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset. Example: `fleurs-r`. ",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device on which a torch.Tensor is or will be allocated",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of waveforms to process at once",
    )

    return parser.parse_args()


def load_audio(pattern, target_sr=SAMPLE_RATE):
    """Load a single audio file as mono waveform at target_sr"""
    wav_path = glob(pattern)
    if len(wav_path) != 1:
        raise ValueError(f"Expected 1 file for {pattern}, got {len(wav_path)}")
    waveform, sr = torchaudio.load(wav_path[0])
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def main():
    """Main loop"""

    args = parse_args()

    # Build dataset with same config as embedding run
    dataset = FleursParallelDataset(
        dataset=args.dataset,
        root_dir=DEFAULT_ROOT_DIR,
        dtype="audio",
        glottocode=None,
        min_speakers=0.0,
    )

    # Load SQUIM objective model
    print("(squim) Loading SQUIM objective model...")
    objective_model = SQUIM_OBJECTIVE.get_model().to(args.device)
    objective_model.eval()

    df = dataset.data.copy().reset_index(drop=True)
    all_records = []
    bs = args.batch_size
    n = len(df)

    with (
        torch.no_grad(),
        torch.amp.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16),
    ):
        for start in tqdm(range(0, n, bs), total=(n + bs - 1) // bs, desc="(squim)"):
            batch = df.iloc[start : start + bs]

            waveforms = []
            for _, row in batch.iterrows():
                _, basename, *_, subset, _ = row
                x = load_audio(f"{dataset.data_dir}/*/*/audio/{subset}/{basename}")
                waveforms.append(x.squeeze(0))

            # Pad to max length in batch
            max_len = max(w.shape[-1] for w in waveforms)
            padded = torch.stack(
                [
                    torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
                    for w in waveforms
                ]
            ).to(args.device)

            stoi, pesq, si_sdr = objective_model(padded)

            for i, (_, row) in enumerate(batch.iterrows()):
                sentence_index, basename, *_, subset, label = row
                all_records.append(
                    {
                        "sentence_index": sentence_index,
                        "language": dataset.label_encoder.decode_ndim(label),
                        "subset": subset,
                        "file": basename,
                        "stoi": stoi[i].item(),
                        "pesq": pesq[i].item(),
                        "si_sdr": si_sdr[i].item(),
                    }
                )

    # Save results alongside the embedding run
    output_file = f"{DEFAULT_METADATA_DIR}/{args.dataset}/squim.csv"
    results = pd.DataFrame(all_records)
    results.to_csv(output_file, index=False)
    print(f"(squim) Saved {len(results)} records to {output_file}")

    # Print per-language summary
    summary = results.groupby("language")[["stoi", "pesq", "si_sdr"]].mean()
    print("\n(squim) Per-language summary (mean):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
