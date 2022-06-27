import pickle
import warnings
from datetime import datetime
from sys import platform

import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
from mrmr import mrmr_classif
from omegaconf import OmegaConf
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def CV_output(scores):
    train_acc_std = round(scores["train_accuracy"].std(), 3)
    test_acc_std = round(scores["test_accuracy"].std(), 3)
    return [
        f"Mean Train Accuracy \t{round(scores['train_accuracy'].mean(), 3)} ({train_acc_std})",
        f"Max Train Accuracy \t\t{round(scores['train_accuracy'].max(), 3)}",
        f"Mean Train Precision \t{round(scores['train_precision'].mean(), 3)}",
        f"Mean Train Recall \t\t{round(scores['train_recall'].mean(), 3)}",
        f"Mean Train F1 Score \t{round(scores['train_f1'].mean(), 3)}",
        f"Mean Test Accuracy \t\t{round(scores['test_accuracy'].mean(), 3)} ({test_acc_std})",
        f"Max Test Accuracy \t\t{round(scores['test_accuracy'].max(), 3)}",
        f"Mean Test Precision \t {round(scores['test_precision'].mean(), 3)}",
        f"Mean Test Recall \t\t {round(scores['test_recall'].mean(), 3)}",
        f"Mean Test F1 Score\t\t {round(scores['test_f1'].mean(), 3)}",
    ]


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
    selected_features = mrmr_classif(X=X, y=y, K=8)

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
    logo = LeaveOneGroupOut()
    gkf = GroupKFold(n_splits=10)

    # Write time and selected features to classification_output.txt for saving results
    write_output = True
    if write_output:
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%y %H:%M:%S")
        with open("classification_output.txt", "a") as o:
            o.write(
                "------ NEW RUN: "
                + dt_string
                + " ------------------------------------------\n"
            )
            o.write("Selected features (" + str(len(selected_features)) + "):\n")
            o.write(str(selected_features) + "\n")
            o.write("\n")

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

        # Print main results
        train_acc_std = round(scores["train_accuracy"].std(), 3)
        test_acc_std = round(scores["test_accuracy"].std(), 3)
        print(name)
        print("-----------------------------------------")
        print(
            f"Mean (std) train accuracy: {round(scores['train_accuracy'].mean(), 3)} ({train_acc_std})"
        )
        print(
            f"Mean (std) test accuracies: {round(scores['test_accuracy'].mean(), 3)} ({test_acc_std})"
        )
        print()

        # Write all results to classification_output.txt
        if write_output:
            output = CV_output(scores)
            with open("classification_output.txt", "a") as o:
                o.write(name + "\n")
                o.write("-----------------------------------------" + "\n")
                for i in output:
                    o.write(i + "\n")
                o.write("\n")

    """
        # print(scores.keys())
        # print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
        # print("Accuracy:", clf.score(X_test, y_test))
        # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        #print(
        #    "Train accuracy: %.3f (%.3f)"
        #    % (np.mean(scores["train_accuracy"]), np.std(scores["train_accuracy"]))
        #)
        #print(
        #    "Test accuracy: %.3f (%.3f)"
        #    % (np.mean(scores["test_accuracy"]), np.std(scores["test_accuracy"]))
        #)

        # final_model = cross_val.best_estimator_
        # clf.fit(X_train, y_train)
        # train_predictions = clf.predict(X_train)
        # val_predictions = final_model.predict(X_val)
        # test_predictions = clf.predict(X_test)

        # print('Train Score:', accuracy_score(train_predictions, y_train)) # .99
        # print('Val Score:', accuracy_score(val_predictions, y_val)) # .89
        # print('Test Score:', accuracy_score(test_predictions, y_test)) # .8


    # Empty array to fill with predictions from each model
    # preds = []

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

    # cm = confusion_matrix(y_test, preds[0])
    # cm_display = ConfusionMatrixDisplay(cm).plot()
    # plt.show()


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
