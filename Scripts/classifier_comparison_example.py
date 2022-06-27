# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import math as m
import pickle
import random
import warnings
from sys import platform

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mrmr import mrmr_classif
from omegaconf import OmegaConf
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import DecisionBoundaryDisplay
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

# Get feature matrix and target vector
current_feature_file = "all_pre_EC_116s"
with open(
    root + "Features_and_output/feature_df_" + current_feature_file + ".pickle",
    "rb",
) as f:
    feature_df = pickle.load(f)

# Create feature matrix and target vector
X = feature_df.iloc[:, :-2]
y = feature_df["Depression"]


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


# ---------------------------- FEATURE SELECTION ---------------------------------

# Choose subset of features for the feature selector to choose from
# sub_cols = [col for col in X.columns if "pow" in col]
# X = X[X.columns.intersection(sub_cols)]

# K is how many features we want to be chosen
# Using MRMR (Minimum Redundancy - Maximum Relevance)
selected_features = mrmr_classif(X=X, y=y, K=2)

# Make feature matrix which has only top K features
chosen_columns = selected_features + ["Subject_ID", "Depression"]
select_features_df = feature_df[feature_df.columns.intersection(chosen_columns)]

# Update feature matrix and target vector
X = select_features_df.iloc[:, :-2].to_numpy()
y = select_features_df["Depression"].to_numpy()
groups = select_features_df["Subject_ID"].to_numpy()

# Normalize X again, since we removed many features
X = zscore(X, axis=None)

# --------------------------------- CLASSIFYING ----------------------------------

names = [
    "Linear discriminant analysis (LDA)",
    "RBF support vector machine (RBFSVM)",
    "Linear support vector machine RBF (LSVM)",
    "K-nearest neighbors (KNN)",
    "Decision tree (DT)",
    "Random forest (RF)",
    "Ensemble classifier",
]

classifiers = [
    LinearDiscriminantAnalysis(),
    SVC(kernel="rbf"),
    SVC(kernel="linear"),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
]

# Add ensemble classifier which combines all the other ones and uses majority voting
estimators = [
    (names[0], classifiers[0]),
    (names[1], classifiers[1]),
    (names[2], classifiers[2]),
    (names[3], classifiers[3]),
    (names[4], classifiers[4]),
    (names[5], classifiers[5]),
]
eclf = VotingClassifier(estimators=estimators, n_jobs=-1, voting="hard")
classifiers.append(eclf)

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

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# ---------------------- PLOTTING ----------------------

figure = plt.figure(figsize=(27, 3))
i = 1

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 0].min() + 0.1

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(1, len(classifiers) + 1, i)
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
# Plot the testing points
ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
)
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=cm, alpha=0.8, ax=ax  # , eps=0.5
    )

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        edgecolors="k",
        alpha=0.6,
    )

    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    # ax.text(
    #    x_max - 0.3,
    #    y_min + 0.3,
    #    ("%.2f" % score).lstrip("0"),
    #    size=15,
    #    horizontalalignment="right",
    # )
    i += 1

plt.tight_layout()
plt.show()
