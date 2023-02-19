from typing import List
import tensorflow as tf
import numpy as np
import pandas as pd
from config import get_logger
logger = get_logger()


def get_mapped_tensor(main_ds: tf.data.Dataset, keys: List[str]):
    return main_ds.map(lambda item: {k: item[k] for k in keys})


def get_integer_lookup_layer(df: pd.DataFrame, key: str):
    ids_vocabulary = tf.keras.layers.IntegerLookup(mask_token=None)
    unique_ids = np.unique(df[key])
    ids_vocabulary.adapt(unique_ids)
    return ids_vocabulary


def get_train_test_split(events_ds: tf.data.Dataset, train_test_split: float):
    """ We split only the purchase events"""
    purchase_events = events_ds.filter(lambda x: x["behavior_type"] == 4)
    other_events = events_ds.filter(lambda x: x["behavior_type"] != 4)
    n_purchases = sum(1 for _ in purchase_events)
    purchase_events_shuffled = purchase_events.shuffle(n_purchases)
    n_test = int(n_purchases * train_test_split)
    n_train = n_purchases - n_test
    purchase_events_train = purchase_events_shuffled.take(n_train)
    purchase_events_test = purchase_events_shuffled.skip(n_train).take(n_test)
    all_events_train = other_events.concatenate(purchase_events_train)
    logger.info(f"Created train and test spits")
    return all_events_train, purchase_events_test


def train_test_split_on_purchases(df: pd.DataFrame, test_ratio: float):
    """

        We are testing on predicting purchases (behavior_type=4). Thus the test set only comprises of behavior_type=4 events
    """
    idx_purchase_events = df[df["behavior_type"] == 4].index.values
    n_test = int(np.floor(test_ratio*len(idx_purchase_events)))
    test_idx = list(np.random.choice(idx_purchase_events, n_test))
    train_idx = set(df.index) - set(test_idx)
    train_df, test_df = df.loc[train_idx], df.loc[test_idx]
    return train_df, test_df
