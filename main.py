"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nik Skouf (sxxxxxxx)
"""

import random

import numpy as np
import pandas as pd

import model_training
import preprocessing

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def normalize_text(text: str) -> str:
    """Return the normalized version of a text"""

    return text.lower()


def main() -> None:
    train, dev, test = preprocessing.preprocessing(RANDOM_SEED)
    train_x, train_y, dev_x, dev_y, test_x, test_y = (
        preprocessing.tfidf_generator(train, dev, test)
    )
    print(train_x, train_y)
    print(dev_x, dev_y)
    print(test_x, test_y)


if __name__ == "__main__":
    main()
