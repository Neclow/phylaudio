"""Download all pre-trained audio models registered in MODEL_ZOO.

Inputs:  none
Options: none
Flow:    Iterate over MODEL_ZOO
                    |
                    v
         Instantiate each extractor + processor (triggers HuggingFace download)
Outputs: model files in the HuggingFace cache directory
"""

from src._config import DEFAULT_CACHE_DIR, SAMPLE_RATE
from src.models._model_zoo import MODEL_ZOO


def main():
    """Main loop"""
    for key, model in MODEL_ZOO.items():
        print(f"Downloading model: {key}")
        extractor = model["extractor"](model_id=key, cache_dir=DEFAULT_CACHE_DIR)
        processor = model["processor"](
            model_id=key, cache_dir=DEFAULT_CACHE_DIR, sr=SAMPLE_RATE
        )
        del extractor, processor


if __name__ == "__main__":
    main()
