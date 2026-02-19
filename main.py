"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nik Skouf (sxxxxxxx)
"""

import random

import numpy as np
import pandas as pd

import preprocessing
import model_training

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# github changed the big file
# train_df = pd.read_json("train.jsonl", lines=True)


def normalize_text(text: str) -> str:
    """Return the normalized version of a text"""

    return text.lower()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
