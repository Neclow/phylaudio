# pylint: disable=invalid-name

"""Base functions for phylogenetic analysis base on the FLEURS dataset."""
import json
import os
import uuid
from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from glob import glob
from typing import List, Optional, Union

import git
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..._config import DEFAULT_EVAL_DIR, DEFAULT_THREADS_NEXUS, MIN_LANGUAGES
from ...data.glottolog import (
    add_language_filter_args,
    filter_languages,
    read_exclude_file,
)
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
    feature_extractor: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None
    parallel_loader: Optional[DataLoader] = None
    classifier: Optional[torch.nn.Module] = None
    decomposer: Optional[torch.nn.Module] = None
    fast_dev_run: bool = False
    # (embeddings, meta) loaded from an extract_embeddings cache; when set, the
    # sentence loop reads pre-computed embeddings instead of running the backbone.
    embedding_cache: Optional[tuple] = None


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
        "--layer",
        type=int,
        default=-1,
        help=(
            "Transformer hidden-state index to use for the embedding. -1 (default) "
            "uses the last layer (last_hidden_state). Only honoured by "
            "wav2vec2/MMS-style extractors."
        ),
    )
    parser.add_argument(
        "--embeddings-cache",
        type=str,
        default=None,
        help=(
            "Path to an extract_embeddings run dir (embeddings.pt + meta.parquet "
            "+ taxa.json). If set, read cached embeddings instead of running the "
            "backbone; language filters apply at read time."
        ),
    )
    add_language_filter_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If true, runs a quick development run for testing purposes",
    )

    return parser


def prepare_everything(args, verbose=True):
    if getattr(args, "discretization", None) == "ste" and args.ckpt is None:
        raise ValueError(
            "--discretization ste requires --ckpt (the trained LID head to "
            "project embeddings through)."
        )

    cfg = vars(args)
    cfg["Commit"] = git.Repo(search_parent_directories=True).head.object.hexsha

    run_id = str(uuid.uuid4())
    cfg["run_id"] = run_id

    if verbose:
        print("Configuration:")
        for k, v in cfg.items():
            print(f"\t{k}: {v}")

    if getattr(args, "embeddings_cache", None):
        return _prepare_from_cache(args, cfg, run_id)

    processor, feature_extractor = prepare_model(args, training=False)

    exclude = read_exclude_file(args.exclude_languages_file)

    parallel_dataset = prepare_dataset(
        args,
        processor=processor,
        split=False,
        fleurs_parallel=True,
        glottocode=args.glottocode,
        min_speakers=args.min_speakers,
        exclude=exclude,
        gender=args.gender,
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


def _prepare_from_cache(args, cfg, run_id):
    """Build a FleursParallelInput from a cached extract_embeddings run.

    Reads pre-computed per-utterance embeddings + aligned meta, applies the
    phylo language filter (glottocode/min_speakers/exclude) to the rows at read
    time while keeping the full label space (matching the live path), and
    attaches the STE head via --ckpt if given. No backbone is loaded.
    """
    cache_dir = args.embeddings_cache

    # Cached embeddings may be float16 (models run half-precision on CUDA); use
    # float32 so the STE projector / downstream run on any device (incl. CPU).
    embeddings = torch.load(f"{cache_dir}/embeddings.pt", map_location="cpu").float()
    meta = pd.read_parquet(f"{cache_dir}/meta.parquet").reset_index(drop=True)
    with open(f"{cache_dir}/taxa.json", "r", encoding="utf-8") as f:
        taxa = json.load(f)

    num_classes = taxa["num_classes"]
    labels = taxa["labels"]

    # Read-time language filter: restrict rows to the analysis languages, keep
    # the full label space (the live FleursParallelDataset filters rows too but
    # does not rebuild the label encoder). Encoded labels map via taxa labels.
    if getattr(args, "glottocode", None) is not None:
        exclude = read_exclude_file(args.exclude_languages_file)
        languages_to_keep = filter_languages(
            args.dataset,
            glottocode=args.glottocode,
            min_speakers=args.min_speakers,
            exclude=exclude,
        )
        name_to_idx = {name: i for i, name in enumerate(labels)}
        keep = {name_to_idx[n] for n in languages_to_keep if n in name_to_idx}
        mask = meta["label"].isin(keep).to_numpy()
        meta = meta[mask].reset_index(drop=True)
        embeddings = embeddings[torch.from_numpy(mask)]
        print(
            f"(cache) {args.glottocode}: kept {len(keep)} languages "
            f"-> {len(meta)} utterances"
        )

    inputs = FleursParallelInput(
        run_id=run_id,
        cfg=cfg,
        num_batches=meta["sentence_index"].nunique(),
        num_classes=num_classes,
        labels=labels,
        fast_dev_run=args.dry_run,
        embedding_cache=(embeddings, meta),
    )

    if args.ckpt is not None:
        inputs.classifier = prepare_classifier(
            args,
            in_dim=embeddings.shape[1],
            out_dim=num_classes,
            dtype=embeddings.dtype,
        )

    return inputs


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


def get_embeddings(fleurs_parallel_input, X, y, device="cpu"):
    """Extract one embedding per utterance.

    Each ``x_i`` is a ``(1, T_i)`` tensor at the utterance's natural length,
    so every extractor runs B=1 over real audio only — no padding, no need
    for attention_mask/lengths/wav_lens plumbing.
    """
    extractor = fleurs_parallel_input.feature_extractor
    all_embeddings = []

    for x_i in tqdm(X, desc="Extracting embeddings", leave=False):
        if isinstance(x_i, list):
            x_i = x_i[0]
        embedding = extractor(x_i.to(device))
        all_embeddings.append(embedding)

    embeddings = torch.cat(all_embeddings, dim=0).to(device)

    return post_process_embeddings(fleurs_parallel_input, embeddings, y)


def post_process_embeddings(fleurs_parallel_input, embeddings, y):
    """Turn raw embeddings into the representation downstream tasks consume.

    Shared by the live (get_embeddings) and cached sentence-loop paths so STE
    projection / classifier filtering / decomposition stay identical.
    """
    if fleurs_parallel_input.classifier is not None:
        if fleurs_parallel_input.cfg.get("discretization") == "ste":
            # Replace each embedding with its STE bottleneck code (hidden_dim-wide, ±1).
            # Match the projector's dtype (the head is cast to the extractor dtype,
            # which can differ from the pooled embedding's, e.g. half vs float).
            projector = fleurs_parallel_input.classifier.projector
            embeddings = projector(
                embeddings.to(next(projector.parameters()).dtype)
            )
        else:
            embeddings, y = filter_embeddings(
                fleurs_parallel_input.classifier, embeddings, y
            )

    if fleurs_parallel_input.decomposer is not None:
        embeddings = decompose(fleurs_parallel_input.decomposer, embeddings)

    return embeddings, y


def sentence_loop(args, inputs, output_folder, downstream_func):
    """Drive ``downstream_func`` per sentence from either the live backbone or a
    cached embedding run (``inputs.embedding_cache``)."""
    if inputs.embedding_cache is not None:
        _sentence_loop_cache(args, inputs, output_folder, downstream_func)
    else:
        _sentence_loop_live(args, inputs, output_folder, downstream_func)


def _sentence_loop_live(args, inputs, output_folder, downstream_func):
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
        sentence_index = batch["sentence_index"][0]

        # Ignore if less than 4 languages ==> cannot build a tree
        if y.unique().shape[0] < MIN_LANGUAGES:
            continue

        with torch.no_grad():
            X_emb, y = get_embeddings(
                fleurs_parallel_input=inputs,
                X=X_input,
                y=y,
                device=args.device,
            )

        downstream_func(X_emb, y, sentence_index, args, inputs, output_folder)

        if args.dry_run:
            break


def _sentence_loop_cache(args, inputs, output_folder, downstream_func):
    """Per-sentence loop reading row-aligned (embeddings, meta) from the cache.

    `meta.parquet` is the linker: group by `sentence_index`, slice the matching
    embedding rows, then apply the same post-processing as the live path.
    """
    embeddings, meta = inputs.embedding_cache

    for sentence_index, grp in tqdm(
        meta.groupby("sentence_index"),
        total=inputs.num_batches,
        desc="(cache) Processing sentence data",
    ):
        y = torch.as_tensor(grp["label"].to_numpy(), device=args.device)

        # Ignore if less than 4 languages ==> cannot build a tree
        if y.unique().shape[0] < MIN_LANGUAGES:
            continue

        X_emb = embeddings[grp.index.to_numpy()].to(args.device)

        with torch.no_grad():
            X_emb, y = post_process_embeddings(inputs, X_emb, y)

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


def resolve_ckpt(ckpt):
    """Accept a checkpoint path or a bare W&B run id (e.g. ``ye1dk63c``).

    A run id is resolved to its checkpoint under
    ``data/eval/<project>/<run_id>/checkpoints/*.ckpt`` (latest if several).
    """
    if os.path.isfile(ckpt):
        return ckpt

    matches = sorted(glob(f"{DEFAULT_EVAL_DIR}/*/{ckpt}/checkpoints/*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"--ckpt '{ckpt}' is neither a file nor a run id with a checkpoint "
            f"under {DEFAULT_EVAL_DIR}/*/{ckpt}/checkpoints/"
        )
    return matches[-1]


def prepare_classifier(args, in_dim, out_dim, dtype):
    ckpt_path = resolve_ckpt(args.ckpt)
    state_dict = torch.load(
        ckpt_path, map_location=args.device, weights_only=False
    )["state_dict"]

    clf_state_dict = {
        k.partition(".")[-1]: v for k, v in state_dict.items() if "classifier" in k
    }

    # Infer the STE-projector (hidden) dim from the checkpoint so its weights
    # load; absent -> linear probe (hidden_dim=None), as before.
    hidden_dim = None
    if "projector.0.weight" in clf_state_dict:
        hidden_dim = clf_state_dict["projector.0.weight"].shape[0]

    classifier = MLP(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)

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
