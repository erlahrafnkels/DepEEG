import pickle
import warnings
from datetime import datetime
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from omegaconf import OmegaConf
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GroupKFold  # , LeaveOneGroupOut
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# import seaborn as sns


warnings.filterwarnings("ignore")
# sys.setrecursionlimit(3 * sys.getrecursionlimit())


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


def plot_features_hist(feature_h, feature_d, bins, title):
    # Plot a histogram comparing a feature between healthy and depressed
    fig = plt.figure(figsize=(10, 6))
    n_d, _, _ = plt.hist(
        x=feature_d,
        bins=bins,
        color=config.colors.dtu_red,
        alpha=0.7,
        rwidth=0.85,
        label="Depressed",
    )
    n_h, _, _ = plt.hist(
        x=feature_h,
        bins=bins,
        color=config.colors.blue,
        alpha=0.7,
        rwidth=0.85,
        label="Healthy",
    )
    plt.grid(axis="y", alpha=0.5)
    plt.xlabel("Value (normalized)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    maxfreq = max(n_d.max(), n_h.max())
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    return fig


def results_to_table(mean, std, pre, rec, f1, par, feat, file):
    # Read and fill in table
    res_df = pd.read_csv("Results/result_table_format.csv")
    res_df["Mean accuracy"] = mean
    res_df["Standard deviation"] = std
    res_df["Precision"] = pre
    res_df["Recall"] = rec
    res_df["F1-score"] = f1
    res_df["Model parameters"] = par
    res_df["Features"] = feat

    # Save table
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%y_%H.%M.%S")
    no_features = str(len(feat[0]))
    filename = "results_" + no_features + "_" + file + "_" + dt_string + ".csv"
    res_df.to_csv("Results/" + filename, sep=",", index=False)

    return


def CV_output(scores):
    # Print main results
    train_acc = round(scores["train_accuracy"].mean(), 3)
    test_acc = round(scores["test_accuracy"].mean(), 3)
    train_std = round(scores["train_accuracy"].std(), 3)
    test_std = round(scores["test_accuracy"].std(), 3)
    print(name)
    print("-----------------------------------------")
    print(f"Mean (std) train acc: {train_acc} ({train_std})")
    print(f"Mean (std) test acc: {test_acc} ({test_std})")
    print()
    return


if __name__ == "__main__":
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

    # ---------------------------- PLOT FEATURES ---------------------------------

    feature_df_h = feature_df[feature_df["Depression"] == 0]
    feature_df_d = feature_df[feature_df["Depression"] == 1]

    mean_cols = [col for col in X.columns if "Mean" in col]
    mean_df_h = feature_df_h[feature_df_h.columns.intersection(mean_cols)].stack()
    mean_df_d = feature_df_d[feature_df_d.columns.intersection(mean_cols)].stack()

    var_cols = [col for col in X.columns if "Var" in col]
    var_df_h = feature_df_h[feature_df_h.columns.intersection(var_cols)].stack()
    var_df_d = feature_df_d[feature_df_d.columns.intersection(var_cols)].stack()

    skew_cols = [col for col in X.columns if "Skew" in col]
    skew_df_h = feature_df_h[feature_df_h.columns.intersection(skew_cols)].stack()
    skew_df_d = feature_df_d[feature_df_d.columns.intersection(skew_cols)].stack()

    kurt_cols = [col for col in X.columns if "Kurt" in col]
    kurt_df_h = feature_df_h[feature_df_h.columns.intersection(kurt_cols)].stack()
    kurt_df_d = feature_df_d[feature_df_d.columns.intersection(kurt_cols)].stack()

    plot_features_hist(mean_df_h, mean_df_d, "auto", "All channel means")
    plot_features_hist(var_df_h, var_df_d, "auto", "All channel variances")
    plot_features_hist(skew_df_h, skew_df_d, "auto", "All channel skewnesses")
    plot_features_hist(kurt_df_h, kurt_df_d, "auto", "All channel kurtoses")
    # plt.show()

    # ---------------------------- FEATURE SELECTION ---------------------------------

    # Choose subset of features for the feature selector to choose from
    # region_names = ["frontal", "temporal", "parietal", "occipital", "central"]
    # X = X.filter(regex='|'.join(region_names))
    # sub_cols = [col for col in X.columns if "pow" in col]
    # X = X[X.columns.intersection(sub_cols)]

    # K is how many features we want to be chosen
    # According to "Statistical challenges of high-dimensional data"
    # BY IAN M. JOHNSTONE AND D. MICHAEL TITTERINGTON a good rule of thumb for how many features
    # to choose is n/p>=5 where n is datapoints and p parameters (features)
    # However, paper also says that a larger number have shown notable success

    # Using MRMR (Minimum Redundancy - Maximum Relevance)
    K = 8
    selected_features = mrmr_classif(X=X, y=y, K=K)

    print("SELECTED FEATURES (", len(selected_features), "):")
    print(selected_features)
    print()

    # Make feature matrix which has only top K features
    chosen_columns = selected_features + ["Subject_ID", "Depression"]
    select_features_df = feature_df[feature_df.columns.intersection(chosen_columns)]

    # Update feature matrix and target vector
    X = select_features_df.iloc[:, :-2].to_numpy()
    y = select_features_df["Depression"].to_numpy()
    groups = select_features_df["Subject_ID"].to_numpy()

    # Correlation between selected features
    # plt.figure()
    # sns.heatmap(select_features_df.iloc[:, :-2].corr())

    # Normalize X again, since we removed many features
    X = zscore(X, axis=None)

    # --------------------------------- CLASSIFYING ----------------------------------

    clf_names = [
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
        (clf_names[0], classifiers[0]),
        (clf_names[1], classifiers[1]),
        (clf_names[2], classifiers[2]),
        (clf_names[3], classifiers[3]),
        (clf_names[4], classifiers[4]),
        (clf_names[5], classifiers[5]),
    ]
    eclf = VotingClassifier(estimators=estimators, n_jobs=-1, voting="hard")
    classifiers.append(eclf)

    # For cross-validation, we use the GROUP functions which will ensure that the same subject
    # can not be present both in train and test sets
    # This matters when/if we will use epoched data
    no_cv_splits = 10
    gkf = GroupKFold(n_splits=no_cv_splits)
    # logo = LeaveOneGroupOut()

    # Initiate lists for accumulating results
    mean_acc_vals = []
    std_vals = []
    precision_vals = []
    recall_vals = []
    f1_score_vals = []
    mdl_param_vals = []
    sel_feat = []

    # Iterate over classifiers
    for name, clf in zip(clf_names, classifiers):
        # Set up the scoring metrics we want to save
        scoring = ["accuracy", "precision", "recall", "f1"]
        # Perform cross-validation
        scores = cross_validate(
            clf,
            X,
            y,
            groups=groups,
            scoring=scoring,
            cv=gkf,
            n_jobs=-1,
            return_train_score=True,
        )

        # Get values for results table
        mean_acc_vals.append(round(scores["test_accuracy"].mean(), 3))
        std_vals.append(round(scores["test_accuracy"].std(), 3))
        precision_vals.append(round(scores["test_precision"].mean(), 3))
        recall_vals.append(round(scores["test_recall"].mean(), 3))
        f1_score_vals.append(round(scores["test_f1"].mean(), 3))
        if name == "Linear discriminant analysis (LDA)":
            sel_feat.append(selected_features)
        else:
            sel_feat.append(None)
        if clf != eclf:
            mdl_param_vals.append(clf.get_params())
        else:
            mdl_param_vals.append(None)

        # Print main results
        CV_output(scores)

    # Results into results table
    save_results = False
    if save_results:
        results_to_table(
            mean_acc_vals,
            std_vals,
            precision_vals,
            recall_vals,
            f1_score_vals,
            mdl_param_vals,
            sel_feat,
            current_feature_file,
        )
