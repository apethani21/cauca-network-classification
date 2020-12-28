import os
import json
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from util import get_train_test_data
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback


BATCH_SIZE = 100000
SEED = 42

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "./logs/" + now + "_test_model"

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


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(50, 50))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", size=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=20)
    plt.yticks(tick_marks, classes, size=20)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 size=20)

    plt.tight_layout()
    plt.ylabel('True label', size=30)
    plt.xlabel('Predicted label', size=30)
    plt.savefig('./test_model/confusion_matrix.png', dpi=250)


def model_builder():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(88,)),
            keras.layers.Dense(256, activation="relu", name="units_0"),
            keras.layers.Dense(512, activation="relu", name="units_1"),
            keras.layers.Dense(768, activation="relu", name="units_2"),
            keras.layers.Dense(512, activation="relu", name="units_3"),
            keras.layers.Dense(50, activation="softmax")
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    return model


def train_and_evaluate():
    os.makedirs("test_model", exist_ok=True)
    X_train, X_test, y_train, y_test = get_train_test_data()
    model = model_builder()
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=50,
        callbacks=[early_stop_callback, tensorboard_callback],
        validation_data=(X_test, y_test),
        use_multiprocessing=True
    )
    model.save("./test_model/model.h5")
    model = keras.models.load_model("./test_model/model.h5")
    with open("./test_model/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred, axis=-1)
    with open("./metadata_preprocessing/protocol_name_encoding.json", "r") as f:
        encoder = json.load(f)
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    cm_df = pd.DataFrame(cm, index=list(encoder), columns=list(encoder))
    cm_df.to_csv("./test_model/confusion_matrix.csv")
    normalised_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    normalised_cm_df = pd.DataFrame(normalised_cm, index=list(encoder), columns=list(encoder))
    normalised_cm_df.to_csv("./test_model/normalised_confusion_matrix.csv")
    plot_confusion_matrix(cm=cm, classes=list(encoder.keys()))
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv("./test_model/clf_report.csv")
    return
    

if __name__ == "__main__":
    train_and_evaluate()
