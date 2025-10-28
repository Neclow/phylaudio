from typing import Final

import torch

RANDOM_STATE: Final = 42

SAMPLE_RATE: Final = 16000

NONE_TENSOR: Final = torch.tensor([-1])

DEFAULT_ROOT_DIR: Final = "data"

DEFAULT_AUDIO_DIR: Final = f"{DEFAULT_ROOT_DIR}/datasets"
DEFAULT_CACHE_DIR: Final = f"{DEFAULT_ROOT_DIR}/models"
DEFAULT_EVAL_DIR: Final = f"{DEFAULT_ROOT_DIR}/eval"
DEFAULT_METADATA_DIR: Final = f"{DEFAULT_ROOT_DIR}/metadata"
DEFAULT_TREE_DIR: Final = f"{DEFAULT_ROOT_DIR}/trees"
DEFAULT_BEAST_DIR: Final = f"{DEFAULT_TREE_DIR}/beast"
DEFAULT_PER_SENTENCE_DIR: Final = f"{DEFAULT_TREE_DIR}/per_sentence"

# Default number of threads for phylogenetic tree inference (iqtree/raxml)
DEFAULT_THREADS_TREE: Final = 4

# Default number for discrete to nexus
DEFAULT_THREADS_NEXUS: Final = 16

# Minimum number of different languages (leaves) to infer a tree
MIN_LANGUAGES: Final = 4
