from utils.common import get_mapped_tensor, get_integer_lookup_layer, get_train_test_split
from config import get_logger, EVENTS_PATH
import pandas as pd
import tensorflow as tf

#  initialize
logger = get_logger(__name__)


def load_raw_data():
    events_df = pd.read_csv(EVENTS_PATH)
    return events_df


def get_dataset_simple():
    events_df = load_raw_data()
    events_ts_ds = tf.data.Dataset.from_tensor_slices(dict(events_df[["user_id", "item_id", "behavior_type"]]))
    logger.info(f"Dataset has {len(events_ts_ds)} events")
    # create mapped tensors
    events_ds = get_mapped_tensor(events_ts_ds, ["user_id", "item_id", "behavior_type"])
    items_ds = get_mapped_tensor(events_ts_ds, ["item_id"])

    return events_df, events_ds, items_ds


def get_dataset_features():
    events_df = load_raw_data()
    # updating/adding time column
    events_df["day_of_week"] = pd.to_datetime(events_df["time"].apply(lambda x: x.split(" ")[0])).dt.dayofweek
    events_df["time"] = events_df["time"].apply(lambda x: int(x.split(" ")[1]))

    events_ts_ds = tf.data.Dataset.from_tensor_slices(
        dict(events_df[["user_id", "item_id", "item_category", "time", "day_of_week", "behavior_type"]]))
    logger.info(f"Dataset has {len(events_ts_ds)} events")
    # create mapped tensors
    events_ds = get_mapped_tensor(events_ts_ds, ["user_id", "item_id", "item_category", "time", "day_of_week", "behavior_type"])
    items_ds = get_mapped_tensor(events_ts_ds, ["item_id", "item_category"])
    return events_df, events_ds, items_ds
