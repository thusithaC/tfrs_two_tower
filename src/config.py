import logging
from datetime import datetime
import sys

PROJECT = "TENSORFLOW_RECOMMENDERS"

USE_MOD_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'

EVENTS_PATH = ["/home/thusitha/work/bigdata/recomendation/data_recomndation/user_behaviour_complete.csv",
               "/home/thusitha/work/projects/recommendation_take_home/data/sample_1k_users.csv",
               "/home/thusitha/work/projects/recommendation_take_home/data/filtered_events.csv"][-1]


def get_logger(name=None):
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    formatter = logging.Formatter(fmt)
    if name is None:
        logger = logging.getLogger(PROJECT)
    else:
        logger = logging.getLogger(name)
    fh = logging.FileHandler(filename=f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_params(model_type):
    params = {
        "defaults": None,
        "train_test_split": 0.10,
        "dense_layers": [32, 32],
        "learning_rate": 0.0005,
        "early_stopping_patience": 3,
        "validation_freq": 100,
        "epochs": 400
    }

    simple_model_params = {
        "embedding_size": 32,
        "batch_size": 4096*2,
        "regularization": 0.001,
        "top_k_accuracy_range": [100],
        "weights": "behavior_type"
    }

    features_model_params = {
        "embedding_size": 32,
        "time_embedding_size": 2,
        "batch_size": 4096 * 2,
        "regularization": 0.001,
        "top_k_accuracy_range": [100],
        "weights": "behavior_type"
    }

    if model_type == "simple":
        params["defaults"] = simple_model_params
    else:
        params["defaults"] = features_model_params

    return params