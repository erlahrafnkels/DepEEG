import math as m
import os
import pickle
import random
from sys import platform

import pandas as pd
from omegaconf import OmegaConf

"""
import numpy as np
import statistics as s
import sklearn

from scipy.stats import skew, kurtosis, zscore
from vmdpy import VMD
"""

# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
samp_freq = config.sample_frequency
noisy_recs = config.noisy_recs

# Get root folder based on which operating system I'm working on
root = "Data/tDCS_EEG_data/Epochs/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root

# List of ready files
recs116s = os.listdir(root + "116_seconds/")
recs116s = sorted(list(set(recs116s) - set(noisy_recs)))

recs116s_EC_pre = [r for r in recs116s if ("EC" in r) and ("pre" in r)]
recs116s_EO_pre = [r for r in recs116s if ("EO" in r) and ("pre" in r)]
recs116s_EC_post = [r for r in recs116s if ("EC" in r) and ("post" in r)]
recs116s_EO_post = [r for r in recs116s if ("EO" in r) and ("post" in r)]

"""
recs116s_y = []
for r in recs116s:
    if "Dep" in r:
        recs116s_y.append([r, 1])
    else:
        recs116s_y.append([r, 0])
"""


def get_subject_id(filename: str) -> int:
    name_split = filename.split("_")
    subject = name_split[0][1:]
    id = int(subject)
    return id


# def make_epochs(record: pd.DataFrame, seconds: int):
#    return record


def split_train_test(recs: list, train_size: float) -> list[list, list]:
    """
    This function only works for un-epoched data (i.e. whole 116 s segments), at least for now.
    For data in smaller segments, we have to make sure data from the same subject can't be present in
    both train and test, only one of them, so there is no information leakage.

    """

    # Start by splitting list into depressed and healthy so we get balanced sets
    d_set = [r for r in recs if "D" in r]
    h_set = [r for r in recs if "H" in r]

    # Because of random, we get new sets each time we run this
    d_train = random.sample(d_set, m.ceil(train_size * len(d_set)))
    d_test = list(set(d_set) - set(d_train))
    h_train = random.sample(h_set, m.ceil(train_size * len(h_set)))
    h_test = list(set(h_set) - set(h_train))

    # Put together
    train = d_train + h_train
    test = d_test + h_test

    # Have only IDs in the lists
    train = [get_subject_id(f) for f in train]
    test = [get_subject_id(f) for f in test]

    return train, test


if __name__ == "__main__":
    # Let's start by looking only at EYES CLOSED AND PRE DATA, since we are working on task 1 (dep or healthy)
    # Split into train and test sets
    # print(recs116s_EC_pre)
    # recs116s_EC_pre_ids = [get_subject_id(f) for f in recs116s_EC_pre]
    # train_ids, test_ids = split_train_test(recs116s_EC_pre, 0.7)

    # Load the data from a pickle file
    with open(root + "116_seconds" + "/all_116s.pickle", "rb") as f:
        all_116s = pickle.load(f)

    # Let's then start by looking at these statistical features:
    # The first four moments of distribution: Mean, variance, skewness and kurtosis
    # We start by taking the whole segment of each signal as it is, per channel
    channel_names = all_116s.columns[:-4]
    id_cols = list(all_116s.columns[-4:])
    mean_names = ["Mean-" + chan for chan in channel_names]
    var_names = ["Var-" + chan for chan in channel_names]
    skew_names = ["Skew-" + chan for chan in channel_names]
    kurt_names = ["Kurt-" + chan for chan in channel_names]
    feature_names = mean_names + var_names + skew_names + kurt_names + id_cols

    # Create empty dataframe for feature matrix
    feature_df = pd.DataFrame(columns=feature_names)

    print(feature_df.columns)

    means = []
    vars = []
    skews = []
    kurts = []

    """
    for id in recs116s_EC_pre_ids:
        print(id)

    for id in train:
        rec = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        for col in rec:
            print(col)
            print(s.mean(rec[col]))

    # Parameters for VMD as proposed from epilepsy paper
    K = np.arange(1, 16, 1)                  # Number of decomposed nodes
    alpha = np.arange(100, 15000, 100)       # Data-fidelity constraint parameter
    tau = 0                                  # Time step of dual ascent
    DC = 0                                   # Number of DC components
    init = 1                                 # Value of initial frequency for the decomposed modes
    tol = 1e-6                               # Tolerance value for convergence criteria
    """
