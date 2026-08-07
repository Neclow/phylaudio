# pylint: disable=redefined-outer-name

import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob

import pandas as pd
import torch
import torchinfo
import wandb
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src._config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_EVAL_DIR,
    SAMPLE_RATE,
)
from src.data import load_dataset
from src.data.datasets import EmbeddingDataset
from src.data.glottolog import (
    add_language_filter_args,
    filter_languages,
    read_exclude_file,
)
from src.models._model_zoo import MODEL_ZOO
from src.tasks.language_identification.classifier import LightningMLP

from pipeline.language_identification import load_cached_datasets

torch.set_float32_matmul_precision("high")

DATASET_ARGS = {
    "dataset": "fleurs-r",
    "dtype": "audio",
    "root_dir": DEFAULT_DATA_DIR,
    "with_vad": False,
}

LOADER_ARGS = {
    "num_workers": 4,
    "batch_size": 64,
    "pin_memory": True,
}

CLF_ARGS = {
    # Number of classes in FLEURS-R
    "num_classes": 102,
    "lr": 2.5e-4,
    "weight_decay": 1e-2,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "project",
        type=str,
        default="phylaudio2",
        help="WandB project name",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Output file (default: {eval_dir}/{project}/summary.csv)",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    add_language_filter_args(parser)
    parser.add_argument(
        "--by",
        type=str,
        default="test_f1",
        choices=["test_accuracy", "test_f1"],
        help="Metric to sort results by (descending)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only process one checkpoint for testing purposes",
    )
    return parser.parse_args()


def _filter_test_dataset(
    test_dataset, glottocode, min_speakers, exclude=None, gender=None, taxa_labels=None
):
    """Filter a test dataset to the analysis language set.

    For EmbeddingDataset, pass taxa_labels (label name list from taxa.json).
    """
    languages_to_keep = filter_languages(
        dataset=DATASET_ARGS["dataset"],
        glottocode=glottocode,
        min_speakers=min_speakers,
        exclude=exclude,
    )

    if taxa_labels is not None:
        labels_to_keep = torch.tensor(
            [i for i, name in enumerate(taxa_labels) if name in languages_to_keep]
        )
        keep = torch.isin(test_dataset.labels, labels_to_keep)
        test_dataset = EmbeddingDataset(
            test_dataset.embeddings[keep], test_dataset.labels[keep]
        )
    else:
        # pylint: disable=unused-variable
        labels_to_keep = test_dataset.label_encoder.encode_sequence(
            languages_to_keep
        )  # noqa: F841
        # pylint: enable=unused-variable
        test_dataset.data = test_dataset.data.query("language in @labels_to_keep")

        if gender is not None:
            mask = test_dataset.data["gender"].str.lower() == gender.lower()
            test_dataset.data = test_dataset.data[mask]
            print(f"Gender filter '{gender}' applied.")

    print(f"Test dataset size after filtering: {len(test_dataset)} samples.")
    return test_dataset


def get_test_data(
    dataset_args, processor, glottocode, min_speakers, exclude=None, gender=None
):
    print("Loading data...")
    _, _, test_dataset = load_dataset(**dataset_args, processor=processor, split=True)
    return _filter_test_dataset(
        test_dataset, glottocode, min_speakers, exclude, gender
    )


def load_embedding_index(embeddings_cache):
    """Build model_id -> cache_dir mapping by scanning cfg.json files."""
    index = {}
    for cfg_path in glob(f"{embeddings_cache}/*/cfg.json"):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("ckpt") is not None:
            continue
        model_id = cfg["model_id"]
        if model_id not in index:
            index[model_id] = str(cfg_path).rsplit("/cfg.json", maxsplit=1)[0]
    return index


if __name__ == "__main__":
    args = parse_args()
    if args.output_file is None:
        args.output_file = f"{DEFAULT_EVAL_DIR}/{args.project}/summary.csv"

    exclude = read_exclude_file(args.exclude_languages_file)

    embeddings_cache = f"{DEFAULT_EMBEDDING_DIR}/{DATASET_ARGS['dataset']}"
    embedding_index = load_embedding_index(embeddings_cache)
    if embedding_index:
        print(f"Embedding cache available for {len(embedding_index)} model(s).")

    api = wandb.Api()

    # Collect runs, cross-check with local checkpoints, group by model_id
    local_run_ids = {
        d for d in os.listdir(f"{DEFAULT_EVAL_DIR}/{args.project}")
        if os.path.isdir(f"{DEFAULT_EVAL_DIR}/{args.project}/{d}")
    }
    runs_by_model = defaultdict(list)
    for run in api.runs(f"phylo2vec/{args.project}", filters={"state": "finished"}):
        if run.id not in local_run_ids:
            continue
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        ckpt_paths = glob(
            f"{DEFAULT_EVAL_DIR}/{args.project}/{run.id}/checkpoints/*.ckpt"
        )
        if not ckpt_paths:
            continue
        runs_by_model[cfg["model_id"]].append((run, cfg, ckpt_paths[0]))

    n_runs = sum(len(v) for v in runs_by_model.values())
    print(f"Found {n_runs} runs across {len(runs_by_model)} model(s).")

    trainer = Trainer(
        devices=[int(args.device.rsplit("cuda:", maxsplit=1)[-1])],
        accelerator="gpu",
        fast_dev_run=args.dry_run,
        logger=False,
    )

    results = {}
    for model_id, model_runs in tqdm(runs_by_model.items()):
        cache_dir = embedding_index.get(model_id)

        if cache_dir is not None:
            print(f"Using cached embeddings for {model_id}...")
            feature_extractor, _, _, test_dataset, _ = load_cached_datasets(cache_dir)
            with open(f"{cache_dir}/taxa.json", "r", encoding="utf-8") as f:
                taxa_labels = json.load(f)["labels"]
            test_dataset = _filter_test_dataset(
                test_dataset,
                glottocode=args.glottocode,
                min_speakers=args.min_speakers,
                exclude=exclude,
                taxa_labels=taxa_labels,
            )
        else:
            # Prepare processor
            print(f"Loading processor for {model_id}...")
            processor_cls = MODEL_ZOO[model_id]["processor"]
            base_kwargs = {
                "model_id": model_id,
                "cache_dir": DEFAULT_CACHE_DIR,
            }
            processor_kwargs = {
                **base_kwargs,
                "sr": SAMPLE_RATE,
            }
            processor = processor_cls(**processor_kwargs)

            # Prepare feature extractor
            print(f"Loading feature extractor for {model_id}...")
            feature_extractor_cls = MODEL_ZOO[model_id]["extractor"]
            feature_extractor_kwargs = {
                **base_kwargs,
                "device": args.device,
                "training": False,
                "finetuned": model_id != "facebook/wav2vec2-xls-r-300m",
            }
            feature_extractor = feature_extractor_cls(**feature_extractor_kwargs)

            test_dataset = get_test_data(
                DATASET_ARGS,
                processor,
                glottocode=args.glottocode,
                min_speakers=args.min_speakers,
                exclude=exclude,
                gender=args.gender,
            )

        test_loader = DataLoader(test_dataset, shuffle=False, **LOADER_ARGS)

        for run, cfg, ckpt_path in model_runs:
            run_id = run.id
            print(f"Loading checkpoint for {run_id} ({model_id})...")
            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            # This fixes a bug in Whisper, not sure why it wasn't working before
            if "whisper" in model_id:
                feature_extractor.dtype = next(
                    iter(ckpt["state_dict"].values())
                ).dtype
            # pylint: disable=no-value-for-parameter
            # NOTE: num_classes might be larger than actual number of classes
            # in filtered data, but that doesn't affect the score calculation
            model = LightningMLP.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                map_location="cpu",
                feature_extractor=feature_extractor,
                loss_fn=ckpt["hyper_parameters"]["loss_fn"],
                hidden_dim=ckpt["hyper_parameters"].get("hidden_dim"),
                dtype=feature_extractor.dtype,
                strict=cache_dir is None,
                **CLF_ARGS,
            )
            # pylint: enable=no-value-for-parameter

            results[run_id] = trainer.test(model=model, dataloaders=test_loader)[0]
            results[run_id]["model_id"] = model_id
            results[run_id]["hidden_dim"] = ckpt["hyper_parameters"].get("hidden_dim")

            model_summary = torchinfo.summary(model)
            results[run_id]["model_size"] = model_summary.total_params

    if args.dry_run:
        print("Dry run enabled, not saving results.")
        print(results)
    else:
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "run_id"
        df.sort_values(args.by, ascending=False, inplace=True)
        df.to_csv(args.output_file)
        print(f"Saved {len(df)} results to {args.output_file}")
