# pylint: disable=invalid-name

"""Base functions for phylogenetic analysis base on the FLEURS dataset."""
import json
import os
import uuid
from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from typing import List, Optional, Union

import git
import joblib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..._config import DEFAULT_THREADS_NEXUS, MIN_LANGUAGES, NONE_TENSOR
from ..common import get_common_args, prepare_dataset, prepare_model
from ..language_identification.classifier import MLP
from ._decomposition import decompose, fit_decomposer

# Data loader default arguments
LOADER_ARGS = {
    "num_workers": 4,
    # Batch size = 1 sentence
    "batch_size": 1,
    "pin_memory": True,
    "shuffle": False,
}


@dataclass
class FleursParallelInput:
    run_id: str
    cfg: dict
    num_batches: int
    num_classes: int
    labels: List[str]
    feature_extractor: Union[torch.nn.Module, List[torch.nn.Module]]
    parallel_loader: DataLoader
    classifier: Optional[torch.nn.Module] = None
    decomposer: Optional[torch.nn.Module] = None
    fast_dev_run: bool = False


def get_fleurs_parallel_args(with_common_args=True):
    """Get arguments for FLEURS sentence-wise analyses

    Parameters
    ----------
    with_common_args : bool, optional
        If True, load common arguments to load data & models, by default True

    Returns
    -------
    parser : argparse.ArgumentParser
        Object for parsing command line strings into Python objects
    """

    def int_or_float(value):
        """
        Custom type function for argparse to accept either an int or a float.
        """
        try:
            # Try converting to an integer first
            return int(value)
        except ValueError:
            try:
                # If int conversion fails, try converting to a float
                value = float(value)
                if 0 < value < 1:
                    return value
                raise ValueError("Float value must be between 0 and 1.")
            except ValueError as err:
                raise ArgumentTypeError(
                    f"'{value}' is not a valid integer or float."
                ) from err

    if with_common_args:
        parser = get_common_args()
    else:
        parser = ArgumentParser()

    parser.add_argument(
        "--ebs",
        type=int,
        default=32,
        help="Feature extraction batch size",
    )
    parser.add_argument(
        "--decomposition",
        type=str,
        help="Whether to decompose the data before discretization (e.g., using PCA)",
    )
    parser.add_argument(
        "-nc",
        "--n-components",
        type=int_or_float,
        help="Number of components to keep after decomposition",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Whether to standardize the data before decomposition",
    )
    parser.add_argument(
        "-nt",
        "--n-threads",
        type=int,
        default=DEFAULT_THREADS_NEXUS,
        help="Number of threads to use for parallel processing of iqtree",
    )
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
        help="If true, runs a quick development run for testing purposes",
    )

    return parser


def prepare_everything(args, verbose=True):
    cfg = vars(args)
    cfg["Commit"] = git.Repo(search_parent_directories=True).head.object.hexsha

    run_id = str(uuid.uuid4())
    cfg["run_id"] = run_id

    if verbose:
        print("Configuration:")
        for k, v in cfg.items():
            print(f"\t{k}: {v}")

    processor, feature_extractor = prepare_model(args, training=False)

    parallel_dataset = prepare_dataset(
        args,
        processor=processor,
        split=False,
        fleurs_parallel=True,
        glottocode=args.glottocode,
        min_speakers=args.min_speakers,
    )[0]

    num_classes = len(parallel_dataset.label_encoder)

    num_batches = len(parallel_dataset)

    labels = parallel_dataset.label_encoder.decode_torch(torch.arange(num_classes))

    parallel_loader = DataLoader(parallel_dataset, **LOADER_ARGS)

    fleurs_parallel_input = FleursParallelInput(
        run_id=run_id,
        cfg=cfg,
        parallel_loader=parallel_loader,
        labels=labels,
        num_batches=num_batches,
        num_classes=num_classes,
        feature_extractor=feature_extractor,
        fast_dev_run=args.dry_run,
    )

    if args.ckpt is not None:
        fleurs_parallel_input.classifier = prepare_classifier(
            args,
            in_dim=feature_extractor.emb_dim,
            out_dim=num_classes,
            dtype=feature_extractor.dtype,
        )

    if args.decomposition is not None:
        fleurs_parallel_input.decomposer = prepare_decomposer(
            args=args,
            fleurs_parallel_input=fleurs_parallel_input,
            sentence_loop_fn=sentence_loop,
        )

    return fleurs_parallel_input


def save_state(fleurs_parallel_input, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(f"{output_folder}/cfg.json", "w", encoding="utf-8") as f:
        json.dump(fleurs_parallel_input.cfg, f, indent=4)

    if fleurs_parallel_input.decomposer is not None:
        joblib.dump(
            fleurs_parallel_input.decomposer,
            f"{output_folder}/_decomposer.pkl",
            compress=True,
        )


def get_embeddings(
    fleurs_parallel_input,
    X,
    y,
    attention_mask,
    device="cpu",
    batch_size=16,
):
    all_embeddings = []

    for i in tqdm(
        range(0, len(X), batch_size), desc="Extracting embeddings", leave=False
    ):
        # X_batch: List
        X_batch = X[i : i + batch_size]

        if isinstance(X_batch[0], list):
            X_batch = [x[0] for x in X_batch]
        else:
            X_batch = torch.stack(X_batch, dim=0).to(device)[:, 0, :]

        # attention_mask: torch.Tensor
        a_batch = attention_mask[i : i + batch_size].to(device)

        if torch.equal(a_batch[0], NONE_TENSOR.to(a_batch.device)):
            embedding = fleurs_parallel_input.feature_extractor(X_batch)
        else:
            embedding = fleurs_parallel_input.feature_extractor(X_batch, a_batch)

        all_embeddings.append(embedding)

    embeddings = torch.cat(all_embeddings, axis=0).to(device)

    if fleurs_parallel_input.classifier is not None:
        embeddings, y = filter_embeddings(
            fleurs_parallel_input.classifier, embeddings, y
        )

    if fleurs_parallel_input.decomposer is not None:
        embeddings = decompose(fleurs_parallel_input.decomposer, embeddings)

    return embeddings, y


def sentence_loop(args, inputs, output_folder, downstream_func):
    for batch in tqdm(
        inputs.parallel_loader,
        total=inputs.num_batches,
        desc="(base) Processing sentence data",
    ):
        # Audio: input shape = N x F*
        # Fasttext: shape = N
        # Non-fasttext: input shape = N x T*
        # N: number of sentences
        # F*: number of frames
        # T*: number of tokens
        X_input = batch["input"]
        y = batch["label"][0].to(args.device)
        attention_mask = batch["attention_mask"][0].to(args.device)
        sentence_index = batch["sentence_index"][0]

        # Ignore if less than 4 languages ==> cannot build a tree
        if y.unique().shape[0] < MIN_LANGUAGES:
            continue

        with torch.no_grad():
            # Audio : embedding shape: N x (C) x D
            # Text: embedding shape: N x (T) x D
            # C: number of chunks
            # T: number of tokens
            # D: embedding dimension
            X_emb, y = get_embeddings(
                fleurs_parallel_input=inputs,
                X=X_input,
                y=y,
                attention_mask=attention_mask,
                batch_size=args.ebs,
                device=args.device,
            )

        downstream_func(X_emb, y, sentence_index, args, inputs, output_folder)

        if args.dry_run:
            break


def filter_embeddings(classifier, X_emb, y):
    y_prob = classifier(X_emb)

    y_pred = y_prob.argmax(dim=-1)

    correct = y.to(X_emb.device) == y_pred

    X_emb = X_emb[correct]

    y = y[correct.to(y.device)]

    return X_emb, y


def prepare_classifier(args, in_dim, out_dim, dtype):
    state_dict = torch.load(args.ckpt, map_location=args.device)["state_dict"]

    clf_state_dict = {
        k.partition(".")[-1]: v for k, v in state_dict.items() if "classifier" in k
    }

    classifier = MLP(in_dim=in_dim, out_dim=out_dim)

    # Strict = False to ignore missing keys (due to prev versions)
    missing_keys, unexpected_keys = classifier.load_state_dict(
        clf_state_dict, strict=False
    )

    print(f"missing keys: {missing_keys}\n" f"unexpec keys: {unexpected_keys}")

    classifier.to(dtype=dtype, device=args.device)

    return classifier


def prepare_decomposer(args, fleurs_parallel_input, sentence_loop_fn):
    print("(base) Entering decomposition loop...")
    all_X_emb = []

    sentence_loop_fn(
        args,
        fleurs_parallel_input,
        output_folder=None,
        downstream_func=lambda x, *args: all_X_emb.append(x),
    )

    X_emb_cat = torch.cat(all_X_emb, dim=0)

    device = args.device[0] if isinstance(args.device, list) else args.device

    decomposer = fit_decomposer(
        X_emb_cat,
        method=args.decomposition,
        n_components=args.n_components,
        standardize=args.standardize,
        device=device,
        seed=args.seed,
    )

    return decomposer
