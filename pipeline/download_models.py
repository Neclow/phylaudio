from src._config import DEFAULT_CACHE_DIR, SAMPLE_RATE
from src.models._model_zoo import MODEL_ZOO

if __name__ == "__main__":
    for key, model in MODEL_ZOO.items():
        print(f"Downloading model: {key}")
        extractor = model["extractor"](model_id=key, cache_dir=DEFAULT_CACHE_DIR)
        processor = model["processor"](
            model_id=key, cache_dir=DEFAULT_CACHE_DIR, sr=SAMPLE_RATE
        )
