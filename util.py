import os
import gc
import numpy as np
import pandas as pd
import logging as lgg
from preprocessing_search import generate_train_test

fmt = "%(asctime)s - %(levelname)s - %(funcName)s:L%(lineno)d - %(message)s"
lgg.basicConfig(format=fmt, level=lgg.INFO)


processing_steps = {
    "log": True,
    "pca": False,
    "smote": False,
    "ros": False,
    "feat": True,
}

def generate_modelling_data():
    os.makedirs("modelling_data", exist_ok=True)

    (rescaled_X_train,
     rescaled_X_test,
     y_train, y_test) = generate_train_test(**processing_steps)

    assert rescaled_X_train.shape[0] == 2853754
    assert rescaled_X_test.shape[0] == 713439
    assert y_train.shape == (2853754,)
    assert y_test.shape == (713439,)

    with open('./modelling_data/model_fit_X_train.npy', 'wb') as f:
        np.save(f, rescaled_X_train)
    del rescaled_X_train
        
    with open('./modelling_data/model_fit_y_train.npy', 'wb') as f:
        np.save(f, y_train.to_numpy())
    del y_train
        
    with open('./modelling_data/X_test.npy', 'wb') as f:
        np.save(f, rescaled_X_test)
    del rescaled_X_test
        
    with open('./modelling_data/y_test.npy', 'wb') as f:
        np.save(f, y_test.to_numpy())
    del y_test
    gc.collect()
            
    df = pd.read_parquet("preprocessed_all.parquet")
    df.drop(y_test.index, inplace=True)
        
    (rescaled_X_train,
     rescaled_X_validation,
     y_train, y_validation) = generate_train_test(df, test_size=0.25, **processing_steps)

    del df; gc.collect()

    assert rescaled_X_train.shape[0] == 2140315
    assert rescaled_X_validation.shape[0] == 713439
    assert y_train.shape == (2140315,)
    assert y_test.shape == (713439,)

    with open('./modelling_data/X_train.npy', 'wb') as f:
        np.save(f, rescaled_X_train)
        
    with open('./modelling_data/y_train.npy', 'wb') as f:
        np.save(f, y_train.to_numpy())
        
    with open('./modelling_data/X_validation.npy', 'wb') as f:
        np.save(f, rescaled_X_validation)
        
    with open('./modelling_data/y_validation.npy', 'wb') as f:
        np.save(f, y_validation.to_numpy())

    return


def get_train_validation_data():
    with open("./modelling_data/X_train.npy", "rb") as f:
        X_train = np.load(f)
    with open("./modelling_data/y_train.npy", "rb") as f:
        y_train = np.load(f)
    with open("./modelling_data/X_validation.npy", "rb") as f:
        X_validation = np.load(f)
    with open("./modelling_data/y_validation.npy", "rb") as f:
        y_validation = np.load(f)
    return X_train, X_validation, y_train, y_validation


def get_train_test_data():
    with open("./modelling_data/model_fit_X_train.npy", "rb") as f:
        model_fit_X_train = np.load(f)
    with open("./modelling_data/model_fit_y_train.npy", "rb") as f:
        model_fit_y_train = np.load(f)
    with open("./modelling_data/X_test.npy", "rb") as f:
        X_test = np.load(f)
    with open("./modelling_data/y_test.npy", "rb") as f:
        y_test = np.load(f)
    return model_fit_X_train, X_test, model_fit_y_train, y_test
