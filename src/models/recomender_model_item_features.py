from typing import Dict, Text, List, Optional
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from utils.common import get_integer_lookup_layer
import pandas as pd
from config import get_params, get_logger

DEFAULTS = get_params("features")["defaults"]


class ItemsModel(tf.keras.Model):

    def __init__(
            self,
            items_vocabulary: tf.keras.layers.IntegerLookup,
            categories_vocabulary: tf.keras.layers.IntegerLookup,
            config: Dict = DEFAULTS):
        super().__init__()

        item_embedding_size = config["embedding_size"]
        category_embedding_size = config["embedding_size"] // 2

        self.id_model = tf.keras.Sequential([
            items_vocabulary,
            tf.keras.layers.Embedding(items_vocabulary.vocabulary_size(), item_embedding_size)
        ])

        self.category_model = tf.keras.Sequential([
            categories_vocabulary,
            tf.keras.layers.Embedding(categories_vocabulary.vocabulary_size(), category_embedding_size)
        ])

        self.final_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config["embedding_size"],
                kernel_regularizer=tf.keras.regularizers.L2(config["regularization"]),
                kernel_initializer=tf.keras.initializers.HeNormal()
            )
        ])

    def call(self, inputs):
        item_embedding = self.id_model(inputs["item_id"])
        category_embedding = self.category_model(inputs["item_category"])
        concat_embedding = tf.concat([item_embedding, category_embedding], 1)
        return self.final_model(concat_embedding)


class UserModel(tf.keras.Model):

    def __init__(
            self,
            user_vocabulary: tf.keras.layers.IntegerLookup,
            time_vocabulary: tf.keras.layers.IntegerLookup,
            dow_ids_vocabulary: tf.keras.layers.IntegerLookup,
            config: Dict = DEFAULTS):
        super().__init__()

        self.id_model = tf.keras.Sequential([
            user_vocabulary,
            tf.keras.layers.Embedding(user_vocabulary.vocabulary_size(), config["embedding_size"])
        ])

        self.time_model = tf.keras.Sequential([
            time_vocabulary,
            tf.keras.layers.Embedding(time_vocabulary.vocabulary_size(), config["time_embedding_size"])
        ])

        self.dow_model = tf.keras.Sequential([
            dow_ids_vocabulary,
            tf.keras.layers.Embedding(dow_ids_vocabulary.vocabulary_size(), config["time_embedding_size"])
        ])

        self.final_model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config["embedding_size"],
                kernel_regularizer=tf.keras.regularizers.L2(config["regularization"]),
                kernel_initializer=tf.keras.initializers.HeNormal()
            )
        ])

    def call(self, inputs):
        user_embedding = self.id_model(inputs["user_id"])
        time_embedding = self.time_model(inputs["time"])
        dow_embedding = self.dow_model(inputs["day_of_week"])
        concat_embedding = tf.concat([user_embedding, time_embedding, dow_embedding], 1)
        return self.final_model(concat_embedding)


class RecommenderModelWithItemFeatures(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate. Under the hood,
    # these are still plain Keras Models.

    def __init__(self,
                 user_ids_vocabulary: tf.keras.layers.IntegerLookup,
                 item_ids_vocabulary: tf.keras.layers.IntegerLookup,
                 category_ids_vocabulary: tf.keras.layers.IntegerLookup,
                 time_ids_vocabulary: tf.keras.layers.IntegerLookup,
                 dow_ids_vocabulary: tf.keras.layers.IntegerLookup,
                 items_ds: tf.data.Dataset,
                 dense_layers: Optional[List] = None,
                 config: Dict = DEFAULTS
                 ):
        super().__init__()

        # Define user and jobs models.

        # query tower
        user_model = tf.keras.Sequential([
            UserModel(user_ids_vocabulary, time_ids_vocabulary, dow_ids_vocabulary)
        ])

        # candidate tower
        items_model = tf.keras.Sequential([
            ItemsModel(item_ids_vocabulary, category_ids_vocabulary)
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

        return self.task(user_embeddings, items_embeddings, compute_metrics=not training)

    @staticmethod
    def get_lookup_layers(df: pd.DataFrame):
        user_ids_vocabulary = get_integer_lookup_layer(df, "user_id")
        item_ids_vocabulary = get_integer_lookup_layer(df, "item_id")
        category_ids_vocabulary = get_integer_lookup_layer(df, "item_category")
        time_ids_vocabulary = get_integer_lookup_layer(df, "time")
        dow_ids_vocabulary = get_integer_lookup_layer(df, "day_of_week")
        return user_ids_vocabulary, item_ids_vocabulary, category_ids_vocabulary, time_ids_vocabulary, dow_ids_vocabulary