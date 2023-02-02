from typing import Dict, Text, Union, List

import numpy as np
import tensorflow as tf
import pandas as pd
from models.recomender_model import RecommenderModel
from models.recomender_model_item_features import RecommenderModelWithItemFeatures
from utils.data_ops import get_dataset_simple, get_dataset_features
from utils.common import get_train_test_split
from config import get_logger, get_params,  EVENTS_PATH

MODEL_TYPE = ["simple", "features"][1]

# initialize
logger = get_logger(__name__)
tf.random.set_seed(42)
np.random.seed(42)

params = get_params(MODEL_TYPE)
logger.info(f"Starting training {MODEL_TYPE}")

if MODEL_TYPE == "simple":
    events_df, events_ds, items_ds = get_dataset_simple()
    # create Integer lookup layers
    user_ids_vocabulary, item_ids_vocabulary = RecommenderModel.get_lookup_layers(events_df)
    # build model
    model = RecommenderModel(user_ids_vocabulary,
                             item_ids_vocabulary,
                             items_ds,
                             dense_layers=params["dense_layers"])
else:
    events_df, events_ds, items_ds = get_dataset_features()
    (user_ids_vocabulary,
     item_ids_vocabulary,
     category_ids_vocabulary,
     time_ids_vocabulary,
     dow_ids_vocabulary) = RecommenderModelWithItemFeatures.get_lookup_layers(events_df)
    model = RecommenderModelWithItemFeatures(user_ids_vocabulary,
                                             item_ids_vocabulary,
                                             category_ids_vocabulary,
                                             time_ids_vocabulary,
                                             dow_ids_vocabulary,
                                             items_ds,
                                             dense_layers=params["dense_layers"]
                                             )

# train and eval
# Crete train test data
train, test = get_train_test_split(events_ds, params["train_test_split"])
cached_train = train.batch(params["defaults"]["batch_size"]).cache()
cached_test = test.batch(params["defaults"]["batch_size"]).cache()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]))
callbacks = [tf.keras.callbacks.EarlyStopping(patience=params["early_stopping_patience"])]

# call the fit method
history = model.fit(cached_train, epochs=params["epochs"],
                    validation_data=cached_test,
                    callbacks=callbacks,
                    validation_freq=params["validation_freq"])

# to get the top k metric
eval_result_train = model.evaluate(cached_train, return_dict=True)
eval_result_test = model.evaluate(cached_test, return_dict=True)

# log results
model_save = {
    "desc": f"all users, {MODEL_TYPE} with dense, buy events only",
    "eval": eval_result_test,
    "eval_train": eval_result_train,
    "data": {"n_train": len(train), "n_test": len(test), "path": EVENTS_PATH},
    "params": params
}

logger.info(model_save)
logger.info("Done!")
