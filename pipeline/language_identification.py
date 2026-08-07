"""End-to-end evaluation of embeddings for audio- or text-based LID"""

import json

import pandas as pd
import torch

from src.data.datasets import EmbeddingDataset
from src.models.embedding import EmbeddingFeatureExtractor
from src.tasks.common import prepare_dataset, prepare_model
from src.tasks.language_identification import fit_predict, parse_lid_args

torch.set_float32_matmul_precision("high")


def load_cached_datasets(cache_dir):
    """Build train/dev/test EmbeddingDatasets + a pass-through extractor from a
    cached extract_embeddings run, so the LID loop runs without the backbone."""
    embeddings = torch.load(f"{cache_dir}/embeddings.pt", map_location="cpu").float()
    meta = pd.read_parquet(f"{cache_dir}/meta.parquet").reset_index(drop=True)
    with open(f"{cache_dir}/taxa.json", "r", encoding="utf-8") as f:
        num_classes = json.load(f)["num_classes"]

    def subset(name):
        mask = (meta["subset"] == name).to_numpy()
        X = embeddings[torch.from_numpy(mask)]
        y = torch.as_tensor(meta.loc[mask, "label"].to_numpy())
        return EmbeddingDataset(X, y)

    feature_extractor = EmbeddingFeatureExtractor(emb_dim=embeddings.shape[1])
    return (
        feature_extractor,
        subset("train"),
        subset("dev"),
        subset("test"),
        num_classes,
    )


def main():
    """Main loop"""
    print("Loading arguments...")
    args = parse_lid_args(with_common_args=True)

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    if args.embeddings_cache:
        feature_extractor, train_dataset, valid_dataset, test_dataset, num_classes = (
            load_cached_datasets(args.embeddings_cache)
        )
    else:
        processor, feature_extractor = prepare_model(args, training=True)

        train_dataset, valid_dataset, test_dataset = prepare_dataset(
            args, processor=processor, split=True
        )

        num_classes = len(train_dataset.label_encoder)

    fit_predict(
        feature_extractor=feature_extractor,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        args=args,
    )


if __name__ == "__main__":
    main()
