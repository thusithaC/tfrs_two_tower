from typing import List
import tensorflow as tf
import numpy as np
import pandas as pd


def get_mapped_tensor(main_ds: tf.data.Dataset, keys: List[str]):
    return main_ds.map(lambda item: {k: item[k] for k in keys})


def get_integer_lookup_layer(df: pd.DataFrame, key: str):
    ids_vocabulary = tf.keras.layers.IntegerLookup(mask_token=None)
    unique_ids = np.unique(df[key])
    ids_vocabulary.adapt(unique_ids)
    return ids_vocabulary


def get_train_test_split(events_ds: tf.data.Dataset, train_test_split: float):
    shuffled_data = events_ds.shuffle(len(events_ds))
    n_train = int(len(events_ds) * train_test_split)
    n_test = len(events_ds) - n_train
    train = shuffled_data.take(n_train)
    test = shuffled_data.skip(n_train).take(n_test)
    print(len(train), len(test))
    return train, test


