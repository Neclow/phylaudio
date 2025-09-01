"""End-to-end evaluation of embeddings for audio- or text-based LID"""

import logging

import torch

from src.tasks.common import prepare_dataset, prepare_model
from src.tasks.language_identification import fit_predict, parse_lid_args

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def main():
    """Main loop"""
    print("Loading arguments...")
    args = parse_lid_args(with_common_args=True)

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

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
