from typing import Dict, Text, List, Optional
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from utils.common import get_integer_lookup_layer
import pandas as pd
from config import get_params, get_logger

DEFAULTS = get_params("simple")["defaults"]


class ItemsModel(tf.keras.Model):

    def __init__(
            self,
            items_vocabulary: tf.keras.layers.IntegerLookup,
            config: Dict = DEFAULTS):
        super().__init__()
        self.id_model = tf.keras.Sequential([
            items_vocabulary,
            tf.keras.layers.Embedding(items_vocabulary.vocabulary_size(), config["embedding_size"])
        ])

    def call(self, inputs):
        return self.id_model(inputs["item_id"])


class UserModel(tf.keras.Model):

    def __init__(
            self,
            user_vocabulary: tf.keras.layers.IntegerLookup,
            config: Dict = DEFAULTS):
        super().__init__()

        self.id_model = tf.keras.Sequential([
            user_vocabulary,
            tf.keras.layers.Embedding(user_vocabulary.vocabulary_size(), config["embedding_size"])
        ])

    def call(self, inputs):
        return self.id_model(inputs["user_id"])


class RecommenderModel(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate. Under the hood,
    # these are still plain Keras Models.

    def __init__(self,
                 user_ids_vocabulary: tf.keras.layers.IntegerLookup,
                 item_ids_vocabulary: tf.keras.layers.IntegerLookup,
                 items_ds: tf.data.Dataset,
                 dense_layers: Optional[List] = None,
                 config: Dict = DEFAULTS
                 ):
        super().__init__()

        # Define user and jobs models.

        # query tower
        user_model = tf.keras.Sequential([
            UserModel(user_ids_vocabulary)
        ])

        # candidate tower
        items_model = tf.keras.Sequential([
            ItemsModel(item_ids_vocabulary)
        ])

        # add layer to both towers if specified

        if dense_layers is not None and len(dense_layers) > 0:
            for i, layer_neurons in enumerate(dense_layers):
                if i == len(dense_layers)-1:
                    activation = None
                else:
                    activation = "relu"

                user_model.add(tf.keras.layers.Dense(
                    layer_neurons,
                    kernel_regularizer=tf.keras.regularizers.L2(config["regularization"]),
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    activation=activation
                    )
                )
                items_model.add(tf.keras.layers.Dense(
                    layer_neurons,
                    kernel_regularizer=tf.keras.regularizers.L2(config["regularization"]),
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    activation=activation
                    )
                )

        # Objective for optimization
        metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=x, name=f"top_{x}_categorical_accuracy") for x in
                   config["top_k_accuracy_range"]]

        # Set up a retrieval task.

        # the retrieval task evaluates the FactorizedTopK metric at the given time with the items embeddings available
        # at that time. i.e. .map accepts a items_model as a parameter, and items_model(...) is called when the
        # evaluation is required
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(candidates=items_ds.batch(config["batch_size"]).map(items_model),
                                                metrics=metrics,
                                                k=config["top_k_accuracy_range"][len(config["top_k_accuracy_range"])-1])
        )

        # Set up user and movie representations.
        self.user_model = user_model
        self.items_model = items_model

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.
        user_embeddings = self.user_model(features)
        items_embeddings = self.items_model(features)

        #  https://github.com/tensorflow/recommenders/issues/140
        #  https: // www.tensorflow.org / recommenders / examples / multitask

        return self.task(user_embeddings, items_embeddings, compute_metrics=not training)

    @staticmethod
    def get_lookup_layers(df: pd.DataFrame):
        user_ids_vocabulary = get_integer_lookup_layer(df, "user_id")
        item_ids_vocabulary = get_integer_lookup_layer(df, "item_id")
        return user_ids_vocabulary, item_ids_vocabulary
