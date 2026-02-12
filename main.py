"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (sxxxxxxx)
Nik Skouf (sxxxxxxx)
"""

import numpy as np
import pandas as pd

# fixed random seed
np.random.seed(42)

test_df = pd.read_json("test.jsonl", lines=True)

# github changed the big file
# train_df = pd.read_json("train.jsonl", lines=True)


def normalize_text(text: str) -> str:
    """Return the normalized version of a text"""

    return text.lower()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
