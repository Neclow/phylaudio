"""Extract a unified frozen-embedding cache for a model.

One per-utterance pass over ALL languages and ALL splits (train/dev/test),
saving raw backbone embeddings aligned with metadata so the same cache serves
both LID (filter on `subset`) and the phylo trees (group on `sentence_index`,
filter to the analysis languages at read time). No language/min-speaker filter
and no MIN_LANGUAGES skip are applied here -- those are read-time concerns.

Output (per run): embeddings.pt (N x D), meta.parquet (label, subset,
sentence_index), labels.pt, taxa.json (full label space), cfg.json.
"""

import json
import os
import uuid
from glob import glob

import git
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src._config import DEFAULT_EMBEDDING_DIR
from src.tasks.common import prepare_dataset, prepare_model
from src.tasks.feature_extraction.base import (
    get_fleurs_parallel_args,
    prepare_classifier,
)

IGNORE_COLUMNS = ["dbs", "ebs", "device", "Commit"]


def update_summary(dataset_embedding_dir):
    """Update summary.csv from all cfg.json files in the embedding directory."""
    cfg_files = glob(f"{dataset_embedding_dir}/*/cfg.json")
    if not cfg_files:
        return

    rows = []
    for cfg_file in cfg_files:
        with open(cfg_file, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    df = (
        pd.DataFrame(rows)
        .set_index("run_id")
        .drop(columns=IGNORE_COLUMNS, errors="ignore")
    )
    fname = "summary.csv"
    df.to_csv(f"{dataset_embedding_dir}/{fname}")
    print(f"Updated {dataset_embedding_dir}/{fname} ({len(df)} runs)")


def parse_args():
    """Parse arguments for per-utterance embedding extraction"""
    parser = get_fleurs_parallel_args(with_common_args=True)

    parser.add_argument(
        "--limit-sentences",
        type=int,
        default=None,
        help="Keep only the first N sentence_indices per split (fast test caches).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = vars(args)
    cfg["Commit"] = git.Repo(search_parent_directories=True).head.object.hexsha
    run_id = str(uuid.uuid4())
    cfg["run_id"] = run_id

    print("Configuration:")
    for k, v in cfg.items():
        print(f"\t{k}: {v}")

    processor, feature_extractor = prepare_model(args, training=False)

    # Per-utterance AudioDatasets for every split (no language filter): the
    # superset both LID and the phylo grouping read from.
    train_dataset, valid_dataset, test_dataset = prepare_dataset(
        args, processor=processor, split=True
    )

    if args.limit_sentences is not None:
        # Keep all languages for the first N sentences per split (fast test cache).
        for ds in (train_dataset, valid_dataset, test_dataset):
            keep = ds.data["sentence_index"].unique()[: args.limit_sentences]
            ds.data = ds.data[ds.data["sentence_index"].isin(keep)].reset_index(
                drop=True
            )

    num_classes = len(train_dataset.label_encoder)
    labels = train_dataset.label_encoder.decode_torch(torch.arange(num_classes))

    dataset_embedding_dir = f"{DEFAULT_EMBEDDING_DIR}/{args.dataset}"
    output_folder = f"{dataset_embedding_dir}/{run_id}"

    os.makedirs(output_folder, exist_ok=True)
    with open(f"{output_folder}/cfg.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    all_emb = []
    records = []

    with torch.no_grad():
        for subset, dataset in (
            ("train", train_dataset),
            ("dev", valid_dataset),
            ("test", test_dataset),
        ):
            # Default collate at batch_size=1 adds the leading batch dim the
            # extractors expect (matches the live get_embeddings path).
            loader = DataLoader(dataset, batch_size=1, num_workers=4)
            for batch in tqdm(loader, desc=f"(extract) {subset}"):
                x = batch["input"]
                if isinstance(x, list):
                    x = x[0]
                emb = feature_extractor(x.to(args.device))
                all_emb.append(emb.cpu())
                records.append(
                    {
                        "label": int(batch["label"][0]),
                        "subset": subset,
                        "sentence_index": str(batch["sentence_index"][0]),
                    }
                )
                if args.dry_run:
                    break

    embeddings = torch.cat(all_emb, dim=0)

    if args.ckpt is not None:
        classifier = prepare_classifier(
            args,
            in_dim=embeddings.shape[1],
            out_dim=num_classes,
            dtype=embeddings.dtype,
        )
        projector = classifier.projector
        with torch.no_grad():
            embeddings = projector[0](embeddings.to(args.device)).cpu()

    meta = pd.DataFrame.from_records(records)

    assert (
        len(meta) == embeddings.shape[0]
    ), f"meta rows ({len(meta)}) != embeddings rows ({embeddings.shape[0]})"

    torch.save(embeddings, f"{output_folder}/embeddings.pt")
    torch.save(torch.as_tensor(meta["label"].to_numpy()), f"{output_folder}/labels.pt")
    meta.to_parquet(f"{output_folder}/meta.parquet")
    with open(f"{output_folder}/taxa.json", "w", encoding="utf-8") as f:
        json.dump({"num_classes": num_classes, "labels": list(labels)}, f, indent=2)

    print(
        f"Saved {embeddings.shape[0]} embeddings (dim {embeddings.shape[1]}) "
        f"to {output_folder}"
    )
    update_summary(dataset_embedding_dir)


if __name__ == "__main__":
    main()
