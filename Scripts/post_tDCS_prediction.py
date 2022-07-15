import pickle
import warnings
from sys import platform

import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from omegaconf import OmegaConf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

warnings.filterwarnings("ignore")


# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
healthy_num = config.subject_classes.healthy_num
depressed_sham_num = config.subject_classes.depressed_sham_num
still_depressed_scores = config.subject_classes.post_dep_depressed_num
noref_post_num = config.subject_classes.noref_post_num
colors = config.colors

# Get root folder based on which operating system I'm working on
root = "Data/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root


if __name__ == "__main__":
    # Best model
    best_model_file = "best_model_EC_ALL_8_LDA_0.858.joblib"

    # Get feature matrix and target vector
    feature_file = root + "tDCS_EEG_data/Features/ALL_feature_df_EC.pickle"
    with open(feature_file, "rb") as f:
        feature_df = pickle.load(f)

    # Pick post data
    feature_df = feature_df[feature_df["Post"] == 1]
    feature_df = feature_df.drop("Post", axis="columns")

    # Remove all healthy subjects and other non-usable
    feature_df = feature_df[~feature_df["Subject_ID"].isin(healthy_num)]
    feature_df = feature_df[~feature_df["Subject_ID"].isin(noref_post_num)]

    # Extract only selected features from best model
    selected_features = [
        "Kurt-FCz",
        "Skew-T4",
        "BIMF1-S-TP8",
        "Var-Fp2",
        "Skew-FT7",
        "Kurt-FC3",
        "Skew-O1",
        "BIMF2-S-T6",
        "Subject_ID",
    ]
    feature_df = feature_df[feature_df.columns.intersection(selected_features)]

    # ----------------------------- TARGET BASED ON ACTIVE/SHAM -----------------------------
    # Make target column based on active/sham - did the subject receive active tDCS or not?
    # If sham -> depression still 1
    # If active -> depression 0
    feature_df_1 = feature_df.copy()
    feature_df_1["Depression"] = feature_df_1.apply(
        lambda row: 1 if row["Subject_ID"] in depressed_sham_num else 0, axis=1
    )

    feature_df_1.to_csv("Results/X_post", sep="\t", index=False)

    # Plot pair plot
    sfdf_plot = feature_df_1.copy()
    sfdf_plot["Subject group"] = sfdf_plot.apply(
        lambda row: "Sham tDCS" if row["Depression"] == 1 else "Active tDCS", axis=1
    )
    pair_plot_colors = {"Sham tDCS": colors.dtu_red, "Active tDCS": colors.blue}

    # Plot pair plot
    sfdf_plot2 = feature_df_1.copy()
    sfdf_plot2["Subject group"] = sfdf_plot2.apply(
        lambda row: "Post-score above 10"
        if row["Depression"] == 1
        else "Post-score below 10",
        axis=1,
    )
    pair_plot_colors2 = {
        "Post-score above 10": colors.dtu_red,
        "Post-score below 10": colors.blue,
    }

    # Create the default pairplot
    sns.pairplot(
        sfdf_plot2,
        vars=feature_df_1.columns[:-2],
        hue="Subject group",
        markers=["o", "D"],
        palette=pair_plot_colors2,
        height=1.9,
        corner=True,
    )
    plt.savefig("Images/pair-plot-post-dass-score_corner.png", dpi=500)
    plt.show()

    # Create feature matrix and target vector
    X1 = feature_df_1.iloc[:, :-2].to_numpy()
    y1 = feature_df_1["Depression"].to_numpy()

    # Get best model
    best_clf = load(root + "Models/" + best_model_file)

    # Predict
    y_pred1 = best_clf.predict(X1)
    acc1 = accuracy_score(y1, y_pred1)
    prec1 = precision_score(y1, y_pred1)
    rec1 = recall_score(y1, y_pred1)
    f11 = f1_score(y1, y_pred1)

    print("TARGET BASED ON ACTIVE/SHAM")
    print("Accuracy:", acc1)
    print("Precision:", prec1)
    print("Recall:", rec1)
    print("F1-score:", f11)
    print()

    # ----------------------------- TARGET BASED ON DASS SCORES -----------------------------
    # Make target column based on active/sham - did the subject receive active tDCS or not?
    # If sham -> depression still 1
    # If active -> depression 0
    feature_df_2 = feature_df.copy()
    feature_df_2["Depression"] = feature_df_2.apply(
        lambda row: 1 if row["Subject_ID"] in still_depressed_scores else 0, axis=1
    )

    # Create feature matrix and target vector
    X2 = feature_df_2.iloc[:, :-2].to_numpy()
    y2 = feature_df_2["Depression"].to_numpy()

    # Subject numbers
    subs2 = feature_df_2["Subject_ID"].to_numpy()

    # Get best model
    best_clf = load(root + "Models/" + best_model_file)

    # Predict
    y_pred2 = best_clf.predict(X2)
    acc2 = accuracy_score(y2, y_pred2)
    prec2 = precision_score(y2, y_pred2)
    rec2 = recall_score(y2, y_pred2)
    f12 = f1_score(y2, y_pred2)

    print("TARGET BASED ON DASS SCORES")
    print("Accuracy:", acc2)
    print("Precision:", prec2)
    print("Recall:", rec2)
    print("F1-score:", f12)
    print()
