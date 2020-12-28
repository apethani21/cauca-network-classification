import os
import gc
import json
import pickle
import logging as lgg
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick import features, target, classifier

from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import ADASYN, SMOTENC, SMOTE, RandomOverSampler

fmt = "%(asctime)s - %(levelname)s - %(funcName)s:L%(lineno)d - %(message)s"
lgg.basicConfig(format=fmt, level=lgg.INFO)

SEED = 42
np.random.seed(SEED)


def add_features(df):
    df["feature1"] = (df["Total.Fwd.Packets"]/(df["Total.Backward.Packets"]+df["Total.Fwd.Packets"])+1)
    df["feature2"] = (df["Fwd.Packet.Length.Std"]/(df["Fwd.Packet.Length.Max"] - df["Fwd.Packet.Length.Min"]+1))**2
    df["feature3"] = df["Total.Length.of.Fwd.Packets"]/(df["Total.Length.of.Bwd.Packets"]+1)
    df["feature4"] = df["Fwd.Packet.Length.Max"]/(df["Fwd.Packet.Length.Min"]+1)
    df["feature5"] = df["Bwd.Packet.Length.Max"]/(df["Bwd.Packet.Length.Min"]+1)
    df["feature6"] = df["Fwd.Packet.Length.Mean"]/(df["Fwd.Packet.Length.Std"]+1)
    df["feature7"] = df["Bwd.Packet.Length.Mean"]/(df["Bwd.Packet.Length.Std"]+1)
    df["feature8"] = df["Fwd.IAT.Total"]/(df["Bwd.IAT.Total"]+1)
    df["feature9"] = ((df["Fwd.IAT.Std"]/(df["Fwd.IAT.Max"] - df["Fwd.IAT.Min"] + 1))**2)
    df["feature10"] = ((df["Bwd.IAT.Std"]/(df["Bwd.IAT.Max"] - df["Bwd.IAT.Min"] + 1))**2)
    df["feature11"] = (df['Flow.Bytes.s']/(df['Flow.Packets.s']+1))*((df['Flow.Bytes.s']/(df['Flow.Packets.s']+1)) - 2)
    df["feature12"] = (df[[col for col in df.columns if col.endswith(".Flag.Count")]]*[10**i for i in range(7)]).sum(axis=1)
    df["feature13"] = np.exp(np.sqrt((df[[col for col in df.columns if col.endswith("Flag.Count")]]).sum(axis=1)))
    df["feature14"] = (df["Subflow.Fwd.Packets"]+df["Subflow.Bwd.Packets"]/(df["Total.Fwd.Packets"]+df["Total.Backward.Packets"]))
    df["feature15"] = (df["Subflow.Fwd.Bytes"]/(df["Subflow.Fwd.Bytes"]+df["Subflow.Bwd.Bytes"]+1))
    df = df[[col for col in df.columns if col not in {"ProtocolName", "target"}]+["ProtocolName", "target"]]
    return df


def _generate_sampling_strategy(df, final_samples_floor=50000):
    target_counts = df["target"].value_counts()
    affected_targets = target_counts[target_counts < final_samples_floor]
    lgg.info(f"over-sampling {len(affected_targets)} classes")
    return {tgt: final_samples_floor for tgt in affected_targets.index}


def generate_train_test(
    df: Optional[pd.DataFrame]=None,
    test_size: float=0.2,
    log: bool=False,
    pca: bool=False,
    adasyn: bool=False,
    smote: bool=False,
    ros: bool=False,
    feat: bool=False
):

    if sum([adasyn, smote, ros]) > 1:
        raise ValueError("Must pick at most 1 minority over-sampling technique.")

    non_feature_cols = ["ProtocolName", "target"]
    if df is None:
        df = pd.read_parquet("preprocessed_all.parquet")
        lgg.info("Dataset loaded")

    with open("./metadata_preprocessing/protocol_name_encoding.json", "r") as f:
        encoder = json.load(f)

    if feat:
        df = add_features(df)
        lgg.info("Added 15 custom features")

    if log:
        for col in df.columns:
            if col in non_feature_cols:
                continue
            else:
                df[col] = np.log(df[col].abs()+1)
        lgg.info("Log transformation applied")
    
    (X_train, X_test,
     y_train, y_test) = train_test_split(df.drop(columns=non_feature_cols),
                                         df["target"],
                                         test_size=test_size,
                                         stratify=df["target"],
                                         random_state=SEED)
    scaler = StandardScaler()
    rescaled_X_train = scaler.fit_transform(X_train)
    rescaled_X_test = scaler.transform(X_test)
    del X_train; del X_test; gc.collect()
    lgg.info("Normalised")

    if pca:
        principal_component_analysis = PCA(random_state=SEED)
        rescaled_X_train = principal_component_analysis.fit_transform(rescaled_X_train)
        rescaled_X_train = pd.DataFrame(rescaled_X_train)
        rescaled_X_train.columns = [f"PCA{i}" for i in range(1, rescaled_X_train.shape[1] + 1)]
        pca_cumsum = principal_component_analysis.explained_variance_ratio_.cumsum()
        components_to_keep = np.argmax(pca_cumsum > 0.975) + 1
        rescaled_X_train = rescaled_X_train[rescaled_X_train.columns[:components_to_keep]]
        rescaled_X_test = pd.DataFrame(principal_component_analysis.transform(rescaled_X_test))
        rescaled_X_test = rescaled_X_test[rescaled_X_test.columns[:components_to_keep]]
        lgg.info(f"Applied PCA keeping {components_to_keep} components")

    if smote:
        sampling_strategy = _generate_sampling_strategy(df)

        if pca:
            sm = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=SEED,
                n_jobs=-1,
                k_neighbors=2
            )
        else:
            rescaled_X_test = pd.DataFrame(rescaled_X_test)
            rescaled_X_test.columns = [col for col in df.columns if col not in non_feature_cols]
            binary_cols = {'Fwd.PSH.Flags', 'FIN.Flag.Count', 'SYN.Flag.Count',
                           'RST.Flag.Count', 'PSH.Flag.Count', 'ACK.Flag.Count',
                           'URG.Flag.Count', 'ECE.Flag.Count', 'Protocol.Other',
                           'Protocol.TCP', 'Protocol.UDP'}
            categorical_features = [i for i, col in enumerate(rescaled_X_test.columns)
                                    if col in binary_cols]
            sm = SMOTENC(
                categorical_features=categorical_features,
                sampling_strategy=sampling_strategy,
                random_state=SEED,
                n_jobs=-1,
                k_neighbors=2
            )
        rescaled_X_train, y_train = sm.fit_resample(rescaled_X_train, y_train)
        lgg.info("Applied SMOTE")

    if adasyn:
        sampling_strategy = _generate_sampling_strategy(df)
        try:
            ada = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=SEED,
                n_jobs=-1,
            )
            rescaled_X_train, y_train = ada.fit_resample(rescaled_X_train, y_train)
            lgg.info("Applied ADASYN")
        except RuntimeError as e:
            lgg.info(f"{e}. Skipping ADASYN transformation.")

    if ros:
        sampling_strategy = _generate_sampling_strategy(df)
        random_over_sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=SEED
        )
        rescaled_X_train, y_train = random_over_sampler.fit_resample(rescaled_X_train, y_train)
        lgg.info("Applied RandomOverSampler")

    return rescaled_X_train, rescaled_X_test, y_train, y_test


def train_and_evaluate(**kwargs):
    (rescaled_X_train,
     rescaled_X_test,
     y_train, y_test) = generate_train_test(**kwargs)

    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        alpha=0.01,
        tol=0.01,
        batch_size=100000,
        learning_rate="adaptive",
        learning_rate_init=0.05,
        max_iter=30,
        n_iter_no_change=5,
        verbose=True,
        random_state=SEED
    )
    lgg.info("Starting MLP Classifier training")
    clf.fit(rescaled_X_train, y_train)
    lgg.info("Finished MLP Classifier training")
    y_pred = clf.predict(rescaled_X_test)

    folder = "_".join([f"{key}_{value}" for key, value in kwargs.items()]).lower()
    output_folder = f"./preprocessing_trials/{folder}"
    os.makedirs(output_folder, exist_ok=True)

    with open(f"{output_folder}/mpl_classifier.pickle", "wb") as f:
        pickle.dump(clf, f)

    experiment_stats = {
        "accurary": 100*((y_test == y_pred).sum())/len(y_pred),
        "loss": list(clf.loss_curve_),
        "train_samples": rescaled_X_train.shape[0],
        "test_samples": rescaled_X_test.shape[0],
        "features": rescaled_X_train.shape[1]
    }

    with open(f"{output_folder}/experiment_stats.json", "w") as f:
        json.dump(experiment_stats, f, indent=4)

    with open("./metadata_preprocessing/protocol_name_encoding.json", "r") as f:
        encoder = json.load(f)
    
    lgg.info("Creating and saving evaluation charts")
    visualizer = classifier.ClassificationReport(
        clf, classes=list(encoder), support=True, 
        is_fitted=True, size=(750, 1000)
    )
    visualizer.fit(rescaled_X_train, y_train)
    visualizer.score(rescaled_X_test, y_test)
    visualizer.show(f"{output_folder}/classification_report.png")

    cm = classifier.ConfusionMatrix(
        clf, classes=list(encoder),
        is_fitted=True, size=(2500, 2500)
    )
    cm.fit(rescaled_X_train, y_train)
    cm.score(rescaled_X_test, y_test)
    cm.show(f"{output_folder}/confusion_matrix.png") 

    cpe_viz = classifier.ClassPredictionError(
        clf, classes=list(encoder),
        is_fitted=True, size=(2000, 2000)
    )
    cpe_viz.fit(rescaled_X_train, y_train)
    cpe_viz.score(rescaled_X_test, y_test)
    cpe_viz.show(f"{output_folder}/class_prediction_error.png")
    return
