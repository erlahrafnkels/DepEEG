import math as m
import pickle
import random
import warnings
from sys import platform

import matplotlib.pyplot as plt
from mrmr import mrmr_classif
# import numpy as np
# import pandas as pd
from omegaconf import OmegaConf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay,  # , accuracy_score
                             confusion_matrix)
from sklearn.model_selection import cross_val_score  # , cross_validate
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
samp_freq = config.sample_frequency
noisy_recs = config.noisy_recs
healthy_num = config.subject_classes.healthy_num
depressed_num = config.subject_classes.depressed_num

# Get root folder based on which operating system I'm working on
root = "Data/tDCS_EEG_data/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root


def split_train_test(recs, train_size):
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

    # Because of random, we get new sets each time we run this
    d_train = random.sample(d_set, m.ceil(train_size * len(d_set)))
    d_test = list(set(d_set) - set(d_train))
    h_train = random.sample(h_set, m.ceil(train_size * len(h_set)))
    h_test = list(set(h_set) - set(h_train))

    # Put together
    train = d_train + h_train
    test = d_test + h_test

    return train, test


if __name__ == "__main__":
    # Get feature matrix and target vector
    with open(
        root + "Features_and_output/feature_df_21.06.22_15:22:03.pickle", "rb"
    ) as f:
        feature_df = pickle.load(f)

    # Create feature matrix and target vector
    X = feature_df.iloc[:, :-2]
    y = feature_df["Depression"]

    # ---------------------------- FEATURE SELECTION ---------------------------------

    # K is how many features we want
    # Using MRMR (Minimum Redundancy - Maximum Relevance)
    selected_features = mrmr_classif(X=X, y=y, K=20)

    # Make feature matrix which has only top K features
    chosen_columns = selected_features + ["Subject_ID", "Depression"]
    select_features_df = feature_df[feature_df.columns.intersection(chosen_columns)]

    # Update feature matrix and target vector
    X = select_features_df.iloc[:, :-2]
    y = select_features_df["Depression"]
    groups = select_features_df["Subject_ID"]

    # ---------------------- SPLITTING INTO TRAIN AND TEST SETS ----------------------

    # Split into train and test
    train, test = split_train_test(select_features_df["Subject_ID"], 0.7)

    # Make train sets
    X_train = select_features_df[select_features_df["Subject_ID"].isin(train)]
    y_train = X_train["Depression"]
    X_train = X_train.iloc[:, :-2]  # Remove ID columns

    # Make test sets
    X_test = select_features_df[select_features_df["Subject_ID"].isin(test)]
    y_test = X_test["Depression"]
    X_test = X_test.iloc[:, :-2]  # Remove ID columns

    # View dimensions
    print("X train shape:\t", X_train.shape)
    print("X test shape:\t", X_test.shape)
    print("y train shape:\t", y_train.shape)
    print("y test shape:\t", y_test.shape)
    print()

    # --------------------------------- CLASSIFYING ----------------------------------

    clf_names = [
        "Linear discriminant analysis (LDA)",
        "Support vector machine (SVM)",
        "K-nearest neighbors (KNN)",
        "Decision tree (DT)",
        "Random forest (RF)",
    ]

    classifiers = [
        LinearDiscriminantAnalysis(),
        SVC(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
    ]

    # For cross-validation, we use the GROUP functions which will ensure that the same subject
    # can not be present both in train and test sets
    # This matters when/if we will use epoched data
    logo = LeaveOneGroupOut()
    gkf = GroupKFold(n_splits=10)

    # Empty array to fill with predictions from each model
    preds = []

    # Iterate over classifiers
    for name, clf in zip(clf_names, classifiers):
        # Evaluate model
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # preds.append(y_pred)
        # scores = cross_validate(clf, X, y, scoring="accuracy", cv=logo, n_jobs=-1, return_train_score=True)
        scores = cross_val_score(clf, X, y, scoring="accuracy", cv=logo, n_jobs=-1)
        # report performance
        # clf_dict["Name"] = name
        # clf_dict["Pred"] = y_pred
        print(name)
        print("Accuracy:", clf.score(X_test, y_test))
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # print('Train accuracy: %.3f (%.3f)' % (np.mean(scores["train_score"]), np.std(scores["train_score"])))
        # print('Test accuracy: %.3f (%.3f)' % (np.mean(scores["test_score"]), np.std(scores["test_score"])))
        print()

    # print(preds)

    cm = confusion_matrix(y_test, preds[0])
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()

    """
    clf.fit(X_train, y_train)
    print(f"----------- {name} -----------")
    print("--- True:")
    print(y_test.to_numpy())
    print("--- Prediction:")
    print(clf.predict(X_test))
    print("--- Accuracy:")
    print(clf.score(X_test, y_test))
    print()
    """
