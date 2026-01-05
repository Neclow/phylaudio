from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from src._config import DEFAULT_CACHE_DIR, SAMPLE_RATE
from src.models._model_zoo import MODEL_ZOO


def parse_args():
    """Parse arguments for downloading audio models"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for key, model in MODEL_ZOO.items():
        print(f"Downloading model: {key}")
        extractor = model["extractor"](model_id=key, cache_dir=DEFAULT_CACHE_DIR)
        processor = model["processor"](
            model_id=key, cache_dir=DEFAULT_CACHE_DIR, sr=SAMPLE_RATE
        )
