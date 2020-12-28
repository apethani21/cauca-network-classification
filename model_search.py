import os
import pickle
import datetime
import logging as lgg
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing_search import generate_train_test
from util import generate_modelling_data, get_train_validation_data
from tensorflow import keras
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
from kerastuner.tuners import Hyperband


BATCH_SIZE = 100000
SEED = 42

log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=10,
    verbose=0
)

tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    embeddings_freq=1,
    write_graph=True,
    update_freq="batch"
)

def shallow_nn_model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(88,)))
    hp_units = hp.Int("units", min_value=256, max_value=1792, step=256)
    model.add(keras.layers.Dense(units=hp_units, activation="elu"))
    model.add(keras.layers.Dense(50, activation="softmax"))
    hp_learning_rate = hp.Choice("learning_rate", values=[0.05, 0.01, 0.0005, 0.0001]) 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model

def deep_dense_nn_model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(88,)))
    for layers in range(hp.Int("num_layers", 2, 8)):
        model.add(
            keras.layers.Dense(units=hp.Int(
                f"units_{layers}",
                min_value=256,
                max_value=2048,
                step=256
            ),
            activation="elu")
        )
    model.add(keras.layers.Dense(50, activation="softmax"))
    hp_learning_rate = hp.Choice("learning_rate", values=[0.0001, 0.0003, 0.0005]) 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model


def long_thin_nn_model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(88,)))
    for layers in range(hp.Int("num_layers", 5, 10)):
        model.add(
            keras.layers.Dense(units=hp.Int(
                f"units_{layers}",
                min_value=256,
                max_value=768,
                step=256
            ),
            activation="elu")
        )
    model.add(keras.layers.Dense(50, activation="softmax"))
    hp_learning_rate = hp.Choice("learning_rate", values=[0.0001, 0.0005, 0.0009])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def long_thin_nn_model_round_two(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(88,)))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(hp.Int(
                "units_2",
                min_value=448,
                max_value=576,
                step=64
            ), activation="relu"))
    model.add(keras.layers.Dense(768, activation="relu"))
    model.add(keras.layers.Dense(hp.Int(
                "units_4",
                min_value=448,
                max_value=576,
                step=64
            ), activation="relu"))
    model.add(keras.layers.Dense(50, activation="softmax"))
    hp_learning_rate = hp.Choice("learning_rate", values=[0.0009, 0.001, 0.0015, 0.002])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def long_thin_nn_model_round_three(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(88,)))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(hp.Int(
                "units_2",
                min_value=384,
                max_value=640,
                step=64
            ), activation="relu"))  # still uncertaintly around this layer, widening search
    model.add(keras.layers.Dense(768, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(50, activation="softmax"))
    hp_learning_rate = hp.Choice("learning_rate", values=[0.0015, 0.002, 0.0025, 0.003])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def run_search(model_builder, directory, project_name):
    os.makedirs(directory, exist_ok=True)
    tuner = RandomSearch(
        model_builder,
        objective="val_accuracy", 
        max_trials=16,
        executions_per_trial=1,
        seed=SEED,
        directory=directory,
        project_name=project_name
    )

    X_train, X_validation, y_train, y_validation = get_train_validation_data()

    tuner.search(
        X_train,
        y_train,
        verbose=1,
        epochs=50,
        batch_size=BATCH_SIZE,
        validation_data=(X_validation, y_validation),
        callbacks=[early_stop_callback, tensorboard_callback],
        use_multiprocessing=True
    )

    tuner.search_space_summary()
    tuner.results_summary()


if __name__ == "__main__":
    run_search(
        long_thin_nn_model_round_three,
        "long_thin_nn_model_round_three",
        "protocol"
    )