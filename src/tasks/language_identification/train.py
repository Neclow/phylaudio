import os
from argparse import ArgumentParser

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from ..._config import DEFAULT_EVAL_DIR
from ..common import get_common_args
from .classifier import LightningMLP


def parse_lid_args(with_common_args=True):
    """Parse arguments for end-to-end LID evaluation"""

    if with_common_args:
        parser = get_common_args()
    else:
        parser = ArgumentParser()

    parser.add_argument("--project", type=str, required=True, help="Wandb project name")
    parser.add_argument("--ext", type=str, default="wav", help="Audio file extension")
    parser.add_argument(
        "--lr",
        type=float,
        default=2.5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3,
        help="Number of training epcohs",
    )
    parser.add_argument("--hidden-dim", type=int, help="Downstream hidden layer size")

    return parser.parse_args()


def fit_predict(
    feature_extractor, train_dataset, valid_dataset, test_dataset, num_classes, args
):
    loader_args = {
        "num_workers": 4,
        "batch_size": args.batch_size,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    print("Preparing downstream classifier...")
    lit_mlp = LightningMLP(
        feature_extractor=feature_extractor,
        num_classes=num_classes,
        loss_fn=nn.NLLLoss(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
    )

    torch.compile(lit_mlp)

    print("Preparing model tracking...")
    os.makedirs(DEFAULT_EVAL_DIR, exist_ok=True)

    # TODO: make it more flexible for multi-GPU
    if "cuda" in args.device:
        accelerator = "gpu"
        devices = [int(args.device.split(":")[-1])]
    else:
        accelerator = "cpu"
        devices = "auto"

    # Logging
    wandb_logger = WandbLogger(project=args.project, save_dir=DEFAULT_EVAL_DIR)

    wandb_logger.experiment.config.update(
        {
            "model_id": args.model_id,
            "finetuned": args.finetuned,
            "max_duration": args.max_length,
        }
    )

    print("Start evaluation!")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.n_epochs,
        enable_model_summary=True,
        callbacks=[
            ModelCheckpoint(
                monitor="valid_loss", mode="min", save_last=False, save_top_k=1
            ),
            TQDMProgressBar(),
        ],
        logger=wandb_logger,
    )

    trainer.fit(
        model=lit_mlp,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    trainer.test(
        model=lit_mlp,
        dataloaders=test_loader,
        ckpt_path="best",
    )
