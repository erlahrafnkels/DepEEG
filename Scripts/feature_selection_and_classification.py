import pickle
import warnings
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from mrmr import mrmr_classif
from omegaconf import OmegaConf

# from seaborn.rcmod import set_style
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
healthy_num = config.subject_classes.healthy_num
depressed_num = config.subject_classes.depressed_num
depressed_sham_num = config.subject_classes.depressed_sham_num
noref_pre_num = config.subject_classes.noref_pre_num
noref_post_num = config.subject_classes.noref_post_num
colors = config.colors

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
    # Set a clean upper y-axis limit
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    return fig


def results_to_table(mean, std, pre, rec, f1, feat, file, rem_corr, y_true, y_pred):
    # Read and fill in table
    res_df = pd.read_csv("Results/result_table_format_v4.csv")
    res_df["Mean"] = mean
    res_df["St.dev."] = std
    res_df["Precision"] = pre
    res_df["Recall"] = rec
    res_df["F1-score"] = f1
    res_df["Remove corr"] = rem_corr
    res_df["Features"] = feat
    res_df["y_true"] = str(y_true)
    res_df["y_pred"] = str(y_pred)

    # Save table
    no_features = str(len(feat[0]))
    max_acc = str(res_df["Mean"].max())
    filename = "results_" + no_features + "_" + file + "-best_" + max_acc + ".csv"
    res_df.to_csv("Results/" + filename, sep=",", index=False)
    print("Results saved")

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
    # ---- VARIABLES ----
    remove_correlated = True
    K_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    plot_pair_plot = True
    save_results = False
    save_model = False

    # ---- COMBINATIONS ----
    # current_feature_file = "all_pre_EC"
    # current_feature_file = "all_pre_EO"
    current_feature_file = "EC"
    # which_features = "stat_"
    # which_features = "pow_chan_"
    # which_features = "pow_reg_"
    # which_features = "vmd_"
    which_features = "ALL_"

    for K in K_range:
        # Get feature matrix and target vector
        with open(
            root
            + "Features/"
            + which_features
            + "feature_df_"
            + current_feature_file
            + ".pickle",
            "rb",
        ) as f:
            feature_df = pickle.load(f)

        # OBS! After re-reviewing the data, it was noticed that some recordings that did not have montage references
        # were still being used - these are removed here
        if current_feature_file == "EC":
            feature_df = feature_df[feature_df["Post"] == 0]
            feature_df = feature_df.drop("Post", axis="columns")
        feature_df = feature_df[~feature_df["Subject_ID"].isin(noref_pre_num)]

        # Create feature matrix and target vector
        X = feature_df.iloc[:, :-2]
        y = feature_df["Depression"]

        # ---------------------------- FEATURE SELECTION ---------------------------------

        # Remove features that are very correlated
        if remove_correlated:
            correlated_features = []
            correlation_matrix = X.corr()

            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        colname = correlation_matrix.columns[i]
                        correlated_features.append(colname)

            print("Before removing correlated:", X.shape)
            X.drop(labels=correlated_features, axis=1, inplace=True)
            print("After removing correlated:", X.shape)
            print()

        # K is how many features we want to be chosen
        # According to "Statistical challenges of high-dimensional data"
        # BY IAN M. JOHNSTONE AND D. MICHAEL TITTERINGTON a good rule of thumb for how many features
        # to choose is n/p>=5 where n is datapoints and p parameters (features)
        # However, paper also says that a larger number have shown notable success

        # Using mRMR (Minimum Redundancy - Maximum Relevance)
        selected_features = mrmr_classif(X=X, y=y, K=K)

        print(f"SELECTED FEATURES ({len(selected_features)}):")
        print(selected_features)
        print()

        # Make feature matrix which has only top K features
        chosen_columns = selected_features + ["Subject_ID", "Depression"]
        select_features_df = feature_df[feature_df.columns.intersection(chosen_columns)]

        # Plot pair plot?
        if plot_pair_plot:
            sfdf_plot = select_features_df.copy()
            sfdf_plot["Subject group"] = sfdf_plot.apply(
                lambda row: "Depressed" if row["Depression"] == 1 else "Healthy", axis=1
            )
            pair_plor_colors = {"Depressed": colors.dtu_red, "Healthy": colors.blue}

            # Create the default pairplot
            sns.pairplot(
                sfdf_plot,
                vars=select_features_df.columns[:-2],
                hue="Subject group",
                markers=["o", "D"],
                palette=pair_plor_colors,
                height=1.9,
                corner=True,
            )
            plt.savefig("Images/pair-plot-best-model-pre_corner.png", dpi=500)
            plt.show()

        print(select_features_df.columns)

        # Update feature matrix and target vector
        X = select_features_df.iloc[:, :-2].to_numpy()
        y = select_features_df["Depression"].to_numpy()
        groups = select_features_df["Subject_ID"].to_numpy()

        # --------------------------------- CLASSIFICATION ON PRE ----------------------------------

        clf_names = [
            "Linear discriminant analysis (LDA)",
            "RBF support vector machine (RBFSVM)",
            "Linear support vector machine RBF (LSVM)",
            "K-nearest neighbors (KNN)",
            "Decision tree (DT)",
            "Ensemble classifier",
        ]

        classifiers = [
            LinearDiscriminantAnalysis(),
            SVC(kernel="rbf"),
            SVC(kernel="linear"),
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
        ]

        # Add ensemble classifier which combines all the other ones and uses majority voting
        estimators = [
            (clf_names[0], classifiers[0]),
            (clf_names[1], classifiers[1]),
            (clf_names[2], classifiers[2]),
            (clf_names[3], classifiers[3]),
            (clf_names[4], classifiers[4]),
        ]
        eclf = VotingClassifier(estimators=estimators, n_jobs=-1, voting="hard")
        classifiers.append(eclf)

        # For cross-validation, we use the GROUP functions which will ensure that the same subject
        # can not be present both in train and test sets
        # This matters when/if we will use epoched data
        no_cv_splits = 10
        gkf = GroupKFold(n_splits=no_cv_splits)

        # Initiate lists for accumulating results
        mean_acc_vals = []
        std_vals = []
        precision_vals = []
        recall_vals = []
        f1_score_vals = []
        sel_feat = []

        # Find best model
        best_clf_acc = 0

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
                return_estimator=True,
            )

            y_train_pred = cross_val_predict(
                clf,
                X,
                y,
                groups=groups,
                cv=gkf,
                n_jobs=-1,
            )

            if name == "Linear discriminant analysis (LDA)":
                y_train_preds = [y_train_pred]
            else:
                y_train_preds = np.append([y_train_preds], [y_train_pred])

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

            # Print main results
            CV_output(scores)

            if save_model:
                test_acc = round(scores["test_accuracy"].mean(), 3)
                if test_acc > best_clf_acc:
                    # Update best
                    best_clf_acc = test_acc

                    # Construct filename
                    name_split = name.split(" ")
                    short_name = name_split[-1]
                    short_name = short_name[1:-1]
                    test_acc = str(test_acc)
                    K_len = str(len(selected_features))
                    filename = (
                        "Data/Models/best_model_"
                        + current_feature_file
                        + "_"
                        + which_features
                        + K_len
                        + "_"
                        + short_name
                        + "_"
                        + test_acc
                        + ".joblib"
                    )

                    # Get model fit (use last model from cv)
                    clf_fit = scores["estimator"][-1]
                    dump(clf_fit, filename)

                    # Display current best model
                    print(f"--- Best model: {short_name}, acc: {test_acc}")
                    print()

        if save_results:
            # Results into results table
            results_to_table(
                mean_acc_vals,
                std_vals,
                precision_vals,
                recall_vals,
                f1_score_vals,
                sel_feat,
                which_features + current_feature_file,
                remove_correlated,
                y,
                y_train_pred,
            )
