from typing import Dict, Text, Union, List

import numpy as np
import tensorflow as tf
import pandas as pd
from models.recomender_model_item_features import DEFAULTS, RecommenderModelWithItemFeatures
from utils.mongo_helper import insert_one
from utils.common import get_mapped_tensor, get_integer_lookup_layer, get_train_test_split

tf.random.set_seed(42)
USE_MOD_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'
EVENTS_PATH = "/home/thusitha/work/projects/recommendation_take_home/data/sample_1k_users.csv"

params = {
    "defaults": DEFAULTS,
    "train_test_split": 0.90,
    "dense_layers": [32, 32],
    "learning_rate": 0.0005,
    "epochs": 100
}


events_dt = pd.read_csv(EVENTS_PATH)
# updating/adding time column
events_dt["day_of_week"] = pd.to_datetime(events_dt["time"].apply(lambda x: x.split(" ")[0])).dt.dayofweek
events_dt["time"] = events_dt["time"].apply(lambda x: int(x.split(" ")[1]))

events_ts_ds = tf.data.Dataset.from_tensor_slices(dict(events_dt[["user_id", "item_id", "item_category", "time", "day_of_week"]]))
print(len(events_ts_ds))

# create mapped tensors
events_ds = get_mapped_tensor(events_ts_ds, ["user_id", "item_id", "item_category", "time", "day_of_week"])
items_ds = get_mapped_tensor(events_ts_ds, ["item_id", "item_category"])

# create Integer lookup layers
user_ids_vocabulary = get_integer_lookup_layer(events_dt, "user_id")
item_ids_vocabulary = get_integer_lookup_layer(events_dt, "item_id")
category_ids_vocabulary = get_integer_lookup_layer(events_dt, "item_category")
time_ids_vocabulary = get_integer_lookup_layer(events_dt, "time")
dow_ids_vocabulary = get_integer_lookup_layer(events_dt, "day_of_week")

# Crete train test data
train, test = get_train_test_split(events_ds, params["train_test_split"])

# build model
model = RecommenderModelWithItemFeatures(user_ids_vocabulary,
                                         item_ids_vocabulary,
                                         category_ids_vocabulary,
                                         time_ids_vocabulary,
                                         dow_ids_vocabulary,
                                         items_ds,
                                         dense_layers=params["dense_layers"]
                                         )

# train and eval
cached_train = train.batch(DEFAULTS["batch_size"]).cache()
cached_test = test.batch(DEFAULTS["batch_size"]).cache()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]))
model.fit(cached_train, epochs=params["epochs"])

eval_result = model.evaluate(cached_test, return_dict=True)

# save results
model_save = {
    "desc": "1 k users, item category and time, dow with dense",
    "eval": eval_result,
    "data": {"n_train": len(train), "n_test": len(test), "path": EVENTS_PATH},
    "params": params
}
print(model_save)
#insert_one(model_save)
