import pandas as pd
from sklearn import model_selection


def preprocessing(random_seed: int):
    test_df = pd.read_json("data/test.jsonl", lines=True)
    train_df = pd.read_json("data/train.jsonl", lines=True)

    train_df, validation_df = model_selection.train_test_split(
        train_df, random_state=random_seed, test_size=0.1, train_size=0.9
    )
    return (train_df, validation_df, test_df)
