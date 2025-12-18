# pylint: disable=redefined-outer-name

import json
from argparse import ArgumentParser
from glob import glob

import torch
import torchinfo
import wandb
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src._config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_EVAL_DIR,
    DEFAULT_ROOT_DIR,
    SAMPLE_RATE,
)
from src.data import load_dataset
from src.data.glottolog import filter_languages_from_glottocode
from src.models._model_zoo import MODEL_ZOO
from src.tasks.language_identification.classifier import LightningMLP

torch.set_float32_matmul_precision("high")

DATASET_ARGS = {
    "dataset": "fleurs-r",
    "dtype": "audio",
    "root_dir": DEFAULT_ROOT_DIR,
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
    "hidden_dim": None,
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--project_name",
        type=str,
        default="phylaudio",
        help="WandB project name",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=f"{DEFAULT_EVAL_DIR}/summary.json",
        help="Output file for the summary results",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--glottocode",
        type=str,
        default="indo1319",
        help="Glottocode to filter languages",
    )
    parser.add_argument(
        "--min-speakers",
        type=float,
        default=1.0,
        help="Minimum number of speakers per language",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only process one checkpoint for testing purposes",
    )
    return parser.parse_args()


def get_test_data(dataset_args, processor, glottocode, min_speakers):
    print("Loading data...")
    _, _, test_dataset = load_dataset(**dataset_args, processor=processor, split=True)

    glottolog_path = f"{test_dataset.meta_dir}/glottolog.csv"
    languages_to_keep = filter_languages_from_glottocode(
        glottolog_path=glottolog_path,
        glottocode=glottocode,
        min_speakers=min_speakers,
    )
    # pylint: disable=unused-variable
    labels_to_keep = test_dataset.label_encoder.encode_sequence(
        languages_to_keep
    )  # noqa: F841
    # pylint: enable=unused-variable
    test_dataset.data = test_dataset.data.query("language in @labels_to_keep")
    print(f"Test dataset size after filtering: {len(test_dataset)} samples.")

    test_loader = DataLoader(test_dataset, shuffle=False, **LOADER_ARGS)
    return test_loader


if __name__ == "__main__":
    args = parse_args()

    ckpts = glob(f"{DEFAULT_EVAL_DIR}/{args.project_name}/*/checkpoints/*.ckpt")
    print(f"Found {len(ckpts)} checkpoints.")

    api = wandb.Api()

    runs = api.runs("phylo2vec/phylaudio")
    results = {}
    for run in tqdm(runs):
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        run_id = run.id
        model_id = cfg["model_id"]

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
            "max_length": MODEL_ZOO[model_id]["max_length"],
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
            "average_pool": False,
        }
        feature_extractor = feature_extractor_cls(**feature_extractor_kwargs)

        print(f"Loading checkpoint for {run_id}...")
        ckpt_path = glob(
            f"{DEFAULT_EVAL_DIR}/{args.project_name}/{run_id}/checkpoints/*.ckpt"
        )[0]
        ckpt = torch.load(ckpt_path, weights_only=False)
        state_dict = ckpt["state_dict"]
        # This fixes a bug in Whisper, not sure why it wasn't working before
        if "whisper" in model_id:
            feature_extractor.dtype = next(iter(state_dict.values())).dtype
        # Load model with checkpoint
        # pylint: disable=no-value-for-parameter
        # NOTE: num_classes might be larger than actual number of classes in filtered data
        # But that shouldn't affect the score calculation
        # Example:
        # >>> from torch import tensor
        # >>> from torchmetrics.classification import MulticlassF1Score
        # >>> metric1 = MulticlassF1Score(num_classes=3, average="macro")
        # >>> metric2 = MulticlassF1Score(num_classes=300, average="macro")
        # >>> preds = tensor([2, 1, 0, 1])
        # >>> target = tensor([2, 1, 0, 0])
        # >>> metric1(preds, target)
        # tensor(0.7778)
        # >>> metric2(preds, target)
        # tensor(0.7778)
        model = LightningMLP.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            feature_extractor=feature_extractor,
            loss_fn=ckpt["hyper_parameters"]["loss_fn"],
            dtype=feature_extractor.dtype,
            **CLF_ARGS,
        )
        # pylint: enable=no-value-for-parameter

        # Prepare test data
        test_loader = get_test_data(
            DATASET_ARGS,
            processor,
            glottocode=args.glottocode,
            min_speakers=args.min_speakers,
        )

        # Calculate test results
        print("Getting test results...")
        trainer = Trainer(
            devices=[int(args.device.rsplit("cuda:", maxsplit=1)[-1])],
            accelerator="gpu",
            fast_dev_run=args.dry_run,
        )
        results[run_id] = trainer.test(model=model, dataloaders=test_loader)[0]

        results[run_id]["model_id"] = model_id

        model_summary = torchinfo.summary(model)
        results[run_id]["model_size"] = model_summary.total_params

    if args.dry_run:
        print("Dry run enabled, not saving results.")
        print(results)
    else:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {args.output_file}")
