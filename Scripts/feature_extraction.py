import math as m
import os
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


def create_combined_dfs():
    # Let's create some combined dataframes
    recs116s_EC_pre = [r for r in recs116s if ("EC" in r) and ("pre" in r)]
    recs116s_EO_pre = [r for r in recs116s if ("EO" in r) and ("pre" in r)]
    recs116s_EC_post = [r for r in recs116s if ("EC" in r) and ("post" in r)]
    recs116s_EO_post = [r for r in recs116s if ("EO" in r) and ("post" in r)]

    # -------------- EC PRE combined dataframe --------------

    # Initialize
    first_EC_pre = recs116s_EC_pre[0]
    first_EC_pre_id = get_subject_id(first_EC_pre)
    first_EC_pre_dep = 1 if "D" in first_EC_pre else 0
    all_pre_EC_116s = pd.read_csv(
        root + "116_seconds/" + first_EC_pre, sep="\t", index_col=False
    )
    all_pre_EC_116s["Subject_ID"] = first_EC_pre_id
    all_pre_EC_116s["Depressed"] = first_EC_pre_dep

    # Then append all other records with the same categorization
    for file in recs116s_EC_pre:
        if file == recs116s_EC_pre[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_pre_EC_116s = all_pre_EC_116s.append(record)

    all_pre_EC_116s.to_csv(
        root + "116_seconds/All_pre_EC_116s.txt", sep="\t", index=False
    )
    print("EC PRE combined dataframe SAVED.")

    # -------------- EO PRE combined dataframe --------------

    # Initialize
    first_EO_pre = recs116s_EO_pre[0]
    first_EO_pre_id = get_subject_id(first_EO_pre)
    first_EO_pre_dep = 1 if "D" in first_EO_pre else 0
    all_pre_EO_116s = pd.read_csv(
        root + "116_seconds/" + first_EO_pre, sep="\t", index_col=False
    )
    all_pre_EO_116s["Subject_ID"] = first_EO_pre_id
    all_pre_EO_116s["Depressed"] = first_EO_pre_dep

    # Then append all other records with the same categorization
    for file in recs116s_EO_pre:
        if file == recs116s_EO_pre[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_pre_EO_116s = all_pre_EO_116s.append(record)

    all_pre_EO_116s.to_csv(
        root + "116_seconds/All_pre_EO_116s.txt", sep="\t", index=False
    )
    print("EO PRE combined dataframe SAVED.")

    # -------------- EC POST combined dataframe --------------

    # Initialize
    first_EC_post = recs116s_EC_post[0]
    first_EC_post_id = get_subject_id(first_EC_post)
    first_EC_post_dep = 1 if "D" in first_EC_post else 0
    all_post_EC_116s = pd.read_csv(
        root + "116_seconds/" + first_EC_post, sep="\t", index_col=False
    )
    all_post_EC_116s["Subject_ID"] = first_EC_post_id
    all_post_EC_116s["Depressed"] = first_EC_post_dep

    # Then append all other records with the same categorization
    for file in recs116s_EC_post:
        if file == recs116s_EC_post[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_post_EC_116s = all_post_EC_116s.append(record)

    all_post_EC_116s.to_csv(
        root + "116_seconds/All_post_EC_116s.txt", sep="\t", index=False
    )
    print("EC POST combined dataframe SAVED.")

    # -------------- EO POST combined dataframe --------------

    # Initialize
    first_EO_post = recs116s_EO_post[0]
    first_EO_post_id = get_subject_id(first_EO_post)
    first_EO_post_dep = 1 if "D" in first_EO_post else 0
    all_post_EO_116s = pd.read_csv(
        root + "116_seconds/" + first_EO_post, sep="\t", index_col=False
    )
    all_post_EO_116s["Subject_ID"] = first_EO_post_id
    all_post_EO_116s["Depressed"] = first_EO_post_dep

    # Then append all other records with the same categorization
    for file in recs116s_EO_post:
        if file == recs116s_EO_post[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_post_EO_116s = all_post_EO_116s.append(record)

    all_post_EO_116s.to_csv(
        root + "116_seconds/All_post_EO_116s.txt", sep="\t", index=False
    )
    print("EO POST combined dataframe SAVED.")

    return


# def make_epochs(record: pd.DataFrame, seconds: int):
#    return record


def split_train_test(recs: list, train_size: float) -> list[list, list]:
    # Start by splitting list into depressed and healthy so we get balanced sets
    d_set = [r for r in recs if "D" in r]
    h_set = [r for r in recs if "H" in r]

    # Because of random, we get new sets each time we run this
    d_train = random.sample(d_set, m.ceil(train_size * len(d_set)))
    d_test = list(set(d_set) - set(d_train))
    h_train = random.sample(h_set, m.ceil(train_size * len(h_set)))
    h_test = list(set(h_set) - set(h_train))

    train = d_train + h_train
    test = d_test + h_test

    return train, test


if __name__ == "__main__":
    # Check whether we have already made and saved the combined data files
    check_file = root + "116_seconds/All_post_EC_116s.txt"
    if not os.path.exists(check_file):
        create_combined_dfs()

    """
    # Let's start by looking only at eyes closed data and pre, since we are working on task 1 (dep or healthy)
    # Split into train and test sets
    train, test = split_train_test(recs116s_EC_pre, 0.7)

    # Let's start by looking at these statistical features:
    # The first four moments of distribution: Mean, variance, skewness and kurtosis
    # We start by taking the whole segment of each signal as it is, per channel
    means = []
    vars = []
    skews = []
    kurts = []

    for file in train:
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
