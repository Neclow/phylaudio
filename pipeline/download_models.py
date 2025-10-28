import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from src._config import DEFAULT_CACHE_DIR, SAMPLE_RATE
from src.models._model_zoo import MODEL_ZOO


def parse_args():
    """Parse arguments for FLEURS-R data preparation"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Output dataset folder. Example: {DEFAULT_CACHE_DIR}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for key, model in MODEL_ZOO.items():
        print(key)
        print(model)
        extractor = model["extractor"](model_id=key, cache_dir=args.output_dir)
        processor = model["processor"](
            model_id=key, cache_dir=args.output_dir, sr=SAMPLE_RATE
        )
