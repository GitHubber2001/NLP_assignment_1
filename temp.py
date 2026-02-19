"""
File to work on the project without merge conflicts
"""

import numpy as np
import pandas as pd
from sklearn import model_selection

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

test_df = pd.read_json("data/test.jsonl", lines=True)
train_df = pd.read_json("data/train.jsonl", lines=True)

print(train_df)
train_df, validation_df = model_selection.train_test_split(
    train_df, random_state=RANDOM_SEED, test_size=0.1, train_size=0.9
)


def normalize_text(text: str) -> str:
    """Return the normalized version of a text"""

    return text.lower()


def main() -> None:
    print(train_df)
    print(validation_df)


if __name__ == "__main__":
    main()
