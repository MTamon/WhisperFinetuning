# import something
from typing import Tuple
import numpy as np
import pandas as pd
from datasets import Dataset, Audio


def data_forming(data_path, save_path):
    # do something
    ...


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


def make_dataset_from_path(data_path, train_ratio=0.7) -> Tuple[Dataset, Dataset]:
    df = load_data(data_path)
    msk = np.random.rand(len(df)) < train_ratio

    train_dataset = (
        Dataset.from_pandas(df[msk])
        .cast_column("path", Audio(sampling_rate=16000))
        .rename_column("path", "audio")
        .remove_columns(["sampling_rate"])
    )
    validate_dataset = (
        Dataset.from_pandas(df[~msk])
        .cast_column("path", Audio(sampling_rate=16000))
        .rename_column("path", "audio")
        .remove_columns(["sampling_rate"])
    )

    return train_dataset, validate_dataset
