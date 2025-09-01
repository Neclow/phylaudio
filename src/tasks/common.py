"""Common functions to load LID data & models"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from lightning.pytorch import seed_everything

from .._config import DEFAULT_CACHE_DIR, DEFAULT_ROOT_DIR, RANDOM_STATE, SAMPLE_RATE
from ..data import load_dataset
from ..models._model_zoo import MODEL_ZOO


def get_common_args():
    """Parse common arguments"""

    parser = ArgumentParser(
        description="Common arguments to load data & models",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="fleurs-r",
        type=str,
        help="Dataset. Example: `fleurs`",
    )
    parser.add_argument(
        "--model_id",
        default="facebook/wav2vec2-xls-r-300m",
        choices=sorted(MODEL_ZOO),
        help="Pre-trained model",
    )
    parser.add_argument(
        "--root-dir",
        default=DEFAULT_ROOT_DIR,
        help="Root folder path for analyses",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Path where cached models are stored",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Number of (audio: number of frames; text: number of tokens)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device on which a torch.Tensor is or will be allocated",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Whether to load a finetuned XLS-R model",
    )
    parser.add_argument(
        "--average-pool",
        action="store_true",
        help=(
            "Whether to perform average pooling on the last hidden state "
            "for `transformers`-based models"
        ),
    )
    parser.add_argument(
        "--with-vad",
        action="store_true",
        help="Whether to transform the data using VAD",
    )
    parser.add_argument("--ckpt", type=str, help="Path to classifier checkpoint")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")

    return parser


def prepare_model(args, training=False):
    seed_everything(args.seed, workers=True)

    base_kwargs = {
        "model_id": args.model_id,
        "cache_dir": args.cache_dir,
    }

    print(f"(common) Loading processor for {args.model_id}...")
    processor_cls = MODEL_ZOO[args.model_id]["processor"]

    if processor_cls is not None:
        processor_kwargs = {
            **base_kwargs,
            "max_length": args.max_length or MODEL_ZOO[args.model_id]["max_length"],
            "sr": SAMPLE_RATE,
        }
        processor = processor_cls(**processor_kwargs)
    else:
        # Fasttext models
        processor = None

    print(f"(common) Loading feature extractor for {args.model_id}...")
    feature_extractor_cls = MODEL_ZOO[args.model_id]["extractor"]

    feature_extractor_kwargs = {
        **base_kwargs,
        "device": args.device,
        "training": training,
        "finetuned": args.finetuned,
        "average_pool": args.average_pool,
    }

    feature_extractor = feature_extractor_cls(**feature_extractor_kwargs)
    feature_extractor.eval()
    feature_extractor.to(args.device)
    torch.compile(feature_extractor)

    return processor, feature_extractor


def prepare_dataset(args, processor, **kwargs):
    print("(common) Preparing data...")
    core_dataset_args = {
        "dataset": args.dataset,
        "dtype": MODEL_ZOO[args.model_id]["dtype"],
        "root_dir": args.root_dir,
        "with_vad": args.with_vad,
        "processor": processor,
    }

    dataset_args = {**core_dataset_args, **kwargs}

    datasets = load_dataset(**dataset_args)

    return datasets
