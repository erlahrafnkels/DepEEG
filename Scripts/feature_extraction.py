import math as m
import pickle
import random
from sys import platform

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import kurtosis, skew, zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from vmdpy import VMD


# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
samp_freq = config.sample_frequency
noisy_recs = config.noisy_recs
healthy_num = config.subject_classes.healthy_num
depressed_num = config.subject_classes.depressed_num

# Get root folder based on which operating system I'm working on
root = "Data/tDCS_EEG_data/Epochs/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root


def get_subject_id(filename: str) -> int:
    name_split = filename.split("_")
    subject = name_split[0][1:]
    id = int(subject)
    return id


# def make_epochs(record: pd.DataFrame, seconds: int):
#    return record


def split_train_test(recs: list, train_size: float):  # -> list[list, list]:
    """
    This function only works for un-epoched data (i.e. whole 116 s segments), at least for now.
    For data in smaller segments, we have to make sure data from the same subject can't be present in
    both train and test, only one of them, so there is no information leakage.

    """

    # Start by splitting list into depressed and healthy so we get balanced sets
    d_set = []
    h_set = []

    for r in recs:
        if r in healthy_num:
            d_set.append(r)
        else:
            h_set.append(r)

    # Start by splitting list into depressed and healthy so we get balanced sets
    # d_set = [r for r in recs if r == 1]  # [r for r in recs if "D" in r]
    # h_set = [r for r in recs if r == 0]  # [r for r in recs if "H" in r]

    # Because of random, we get new sets each time we run this
    d_train = random.sample(d_set, m.ceil(train_size * len(d_set)))
    d_test = list(set(d_set) - set(d_train))
    h_train = random.sample(h_set, m.ceil(train_size * len(h_set)))
    h_test = list(set(h_set) - set(h_train))

    # Put together
    train = d_train + h_train
    test = d_test + h_test

    # Have only IDs in the lists
    # train = [get_subject_id(f) for f in train]
    # test = [get_subject_id(f) for f in test]

    return train, test


if __name__ == "__main__":
    # Let's start by looking only at EYES CLOSED AND PRE DATA, since we are working on task 1 (dep or healthy)
    # Load the data from a pickle file
    with open(root + "116_seconds" + "/all_pre_EC_116s.pickle", "rb") as f:
        all_pre_EC_116s = pickle.load(f)

    # Let's then start by looking at these statistical features:
    # The first four moments of distribution: Mean, variance, skewness and kurtosis
    # We start by taking the whole segment of each signal as it is, per channel
    # Create columns for feature matrix
    channel_names = all_pre_EC_116s.columns[:-2]
    mean_names = ["Mean-" + chan for chan in channel_names]
    var_names = ["Var-" + chan for chan in channel_names]
    skew_names = ["Skew-" + chan for chan in channel_names]
    kurt_names = ["Kurt-" + chan for chan in channel_names]
    feature_names = mean_names + var_names + skew_names + kurt_names

    # Create empty lists for feature matrix
    feature_mat = []
    targets = []

    # List of subject numbers present in data table
    subject_ids = all_pre_EC_116s["Subject_ID"].unique()

    # Iterate over all subjects and get the stats for each channel, for each subject
    for sub in subject_ids:
        # Get subtable for subject and depression value
        current_sub = all_pre_EC_116s[all_pre_EC_116s["Subject_ID"] == sub]
        is_depressed = current_sub["Depressed"].iloc[0]
        current_sub = current_sub.to_numpy()
        current_sub = np.delete(current_sub, [30, 31], axis=1)

        # Calculate statistics per channel, these give (30,) arrays
        means = np.mean(current_sub, axis=0)
        vars = np.var(current_sub, axis=0)
        skews = skew(current_sub, axis=0)
        kurts = kurtosis(current_sub, axis=0)

        # Concatenate arrays to make the whole row for the subject, for inserting into the feature matrix
        feature_row = np.concatenate((means, vars, skews, kurts))
        targets.append([sub, is_depressed])
        feature_mat.append(feature_row)

    # When feature matrix has been filled with values, we normalize it
    feature_mat = np.array(feature_mat)
    feature_mat = zscore(feature_mat, axis=None)
    targets = np.array(targets)

    # Then put it into a dataframe so we have the column names
    feature_df = pd.DataFrame(feature_mat, columns=feature_names)
    feature_df["Subject_ID"] = targets[:, 0]
    feature_df["Depression"] = targets[:, 1]

    # Split into train and test
    train, test = split_train_test(targets[:, 0], 0.7)
    X_train = feature_df[feature_df["Subject_ID"].isin(train)]
    y_train = X_train["Depression"]
    X_train = X_train.iloc[:, :-2]
    X_test = feature_df[feature_df["Subject_ID"].isin(test)]
    y_test = X_test["Depression"]
    X_test = X_test.iloc[:, :-2]

    print("X train shape:\t", X_train.shape)
    print("X test shape:\t", X_test.shape)
    print("y train shape:\t", y_train.shape)
    print("y test shape:\t", y_test.shape)

    # Try first classifier!!!
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)
    print("--- TRUE ---")
    print(y_test)
    print("--- PREDICTION ---")
    print(LDA.predict(X_test))
    print("--- ACCURACY ---")
    print(LDA.score(X_test, y_test))

    """
    # ------------------------------ MODEL WITH NUMPY ------------------------------
    # And convert back to numpy for the classification
    # --- I know going back and forth is a little stupid but I don't have time to optimize :') ---
    X_train = features_train.to_numpy()
    X_test = features_test.to_numpy()
    y_train = targets_train.to_numpy()
    y_test = targets_test.to_numpy()

    print("X train shape:\t", X_train.shape)
    print("X test shape:\t", X_test.shape)
    print("y train shape:\t", y_train.shape)
    print("y test shape:\t", y_test.shape)

    # Try first classifier!!!
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)
    print("--- TRUE ---")
    print(y_test)
    print("--- PREDICTION ---")
    print(LDA.predict(X_test))
    print("--- ACCURACY ---")
    print(LDA.score(X_test, y_test))

    # ------------------------------ VMD STUFF------------------------------

    # Parameters for VMD as proposed from epilepsy paper
    K = np.arange(1, 16, 1)                  # Number of decomposed nodes
    alpha = np.arange(100, 15000, 100)       # Data-fidelity constraint parameter
    tau = 0                                  # Time step of dual ascent
    DC = 0                                   # Number of DC components
    init = 1                                 # Value of initial frequency for the decomposed modes
    tol = 1e-6                               # Tolerance value for convergence criteria
    """
