import math as m
import pickle
import random
import warnings
from sys import platform

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import kurtosis, skew, zscore

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from vmdpy import VMD

warnings.filterwarnings("ignore")


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


def get_subject_id(filename):
    name_split = filename.split("_")
    subject = name_split[0][1:]
    id = int(subject)
    return id


def make_epochs(df, seconds, overlap):
    # Input df: Table with all 116s for pre/post and EO/EC
    subjects = df["Subject_ID"].unique()
    step = int(seconds * overlap)
    end = int(
        (116 // seconds) * seconds + step
    )  # Fix/check if i make different epochs than 10s!!
    cuts = np.arange(0, end * samp_freq, step * samp_freq)
    all_pre_EC_10s_epochs = pd.DataFrame(columns=df.columns)
    all_pre_EC_10s_epochs["Epoch"] = None

    s = 1
    for subject in subjects:
        current_sub = df[df["Subject_ID"] == subject]
        epoch = 0
        print(f"Making epochs for subject {s}/{subjects.shape[0]}")
        for i in range(len(cuts[:-2])):
            start = cuts[i]
            end = cuts[i + 2]
            current_epoch = current_sub.iloc[start:end, :]
            current_epoch["Epoch"] = epoch
            all_pre_EC_10s_epochs = all_pre_EC_10s_epochs.append(current_epoch)
            epoch += 1
        s += 1

    # Save table as .pickle file
    with open(root + "10_seconds" + "/all_pre_EC_10s_epochs.pickle", "wb") as f:
        pickle.dump(all_pre_EC_10s_epochs, f)

    return


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

    print(feature_df.shape)

    # ------------------------------------------------ VMD STUFF ------------------------------------------------

    """
    PAPER: Epilepsy seizure detection using kurtosis based VMD’s parameters selection and bandwidth features
    AUTHORS: Sukriti, Monisha Chakraborty, Debjani Mitra

    The proposed methodology is described as follows:

    Step 1: A range of K and α is set such that K = 1–15 with an interval of 1 and α = 100–15000 with an interval
            of 100.
    Step 2: A new signal is formulated such that it consists of 10 s of EEG segments from each of the datasets
            Z, F and S.
    Step 3: The new signal is decomposed by VMD for all possible combinations of K and α to obtain K BIMFs.
    Step 4: The BIMFs are then summed up to obtain the reconstruction signal for each decomposition.
    Step 5: Kurtosis of each reconstructed signal is determined.
    Step 6: The value of K and α for which kurtosis is maximum is recorded.
    Step 7: Lastly, the five sets of EEG data (Z, O, N, F and S) are decomposed into BIMFs by VMD under the
            recorded value of K and α.
    """

    # Step 1: Parameters for VMD proposed in the paper
    K = np.arange(1, 16, 1)  # Number of decomposed nodes
    alpha = np.arange(100, 15100, 100)  # Data-fidelity constraint parameter
    tau = 0  # Time step of dual ascent
    DC = 0  # Number of DC components
    init = 1  # Value of initial frequency for the decomposed modes
    tol = 1e-6  # Tolerance value for convergence criteria

    # Step 2: Split data into 10-second epochs
    # make_epochs(all_pre_EC_116s, 10, 0.5) --> Already done, comment out
    with open(root + "10_seconds" + "/all_pre_EC_10s_epochs.pickle", "rb") as f:
        all_pre_EC_10s_epochs = pickle.load(f)

    # Step 3: Run VMD
    # Outputs:
    # u       - the collection of decomposed modes/BIMFs
    # u_hat   - spectra of the modes
    # omega   - estimated mode center-frequencies

    epochs = all_pre_EC_10s_epochs["Epoch"].unique()
    max_kurt = 0

    for k in K:
        for a in alpha:
            kurt = random.randint(0, 25)
            if kurt > max_kurt:
                max_kurt = kurt
            print(f"{k} \t {a} \t {max_kurt}")

    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

    """
    # ------------------------------------------------ CLASSIFYING ------------------------------------------------

    # Try first classifier!!!
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)
    print("--- TRUE ---")
    print(y_test.to_numpy())
    print("--- PREDICTION ---")
    print(LDA.predict(X_test))
    print("--- ACCURACY ---")
    print(LDA.score(X_test, y_test))

    # ------------------------------------------------ PLOTTING ------------------------------------------------

    mean_df = feature_df[feature_df.columns.intersection(mean_names + ["Depression"])]
    var_df = feature_df[feature_df.columns.intersection(var_names + ["Depression"])]
    skew_df = feature_df[feature_df.columns.intersection(skew_names + ["Depression"])]
    kurt_df = feature_df[feature_df.columns.intersection(kurt_names + ["Depression"])]

    mean_df_dep = mean_df[mean_df["Depression"] == 1].iloc[:, :-1]
    mean_df_hel = mean_df[mean_df["Depression"] == 0].iloc[:, :-1]
    var_df_dep = var_df[var_df["Depression"] == 1].iloc[:, :-1]
    var_df_hel = var_df[var_df["Depression"] == 0].iloc[:, :-1]
    skew_df_dep = skew_df[skew_df["Depression"] == 1].iloc[:, :-1]
    skew_df_hel = skew_df[skew_df["Depression"] == 0].iloc[:, :-1]
    kurt_df_dep = kurt_df[kurt_df["Depression"] == 1].iloc[:, :-1]
    kurt_df_hel = kurt_df[kurt_df["Depression"] == 0].iloc[:, :-1]

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].scatter(mean_df_dep, var_df_dep, label="D", s=5, color=config.colors.dtu_red)
    axs[0, 0].scatter(mean_df_hel, var_df_hel, label="H", s=5, color=config.colors.bright_green)
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].set_title("Mean vs. variance")

    axs[0, 1].scatter(mean_df_dep, skew_df_dep, label="D", s=5, color=config.colors.dtu_red)
    axs[0, 1].scatter(mean_df_hel, skew_df_hel, label="H", s=5, color=config.colors.bright_green)
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_title("Mean vs. skewness")

    axs[0, 2].scatter(mean_df_dep, kurt_df_dep, label="D", s=5, color=config.colors.dtu_red)
    axs[0, 2].scatter(mean_df_hel, kurt_df_hel, label="H", s=5, color=config.colors.bright_green)
    axs[0, 2].legend(loc="upper right")
    axs[0, 2].set_title("Mean vs. kurtosis")

    axs[1, 0].scatter(var_df_dep, skew_df_dep, label="D", s=5, color=config.colors.dtu_red)
    axs[1, 0].scatter(var_df_hel, skew_df_hel, label="H", s=5, color=config.colors.bright_green)
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].set_title("Variance vs. skewness")

    axs[1, 1].scatter(var_df_dep, kurt_df_dep, label="D", s=5, color=config.colors.dtu_red)
    axs[1, 1].scatter(var_df_hel, kurt_df_hel, label="H", s=5, color=config.colors.bright_green)
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].set_title("Variance vs. kurtosis")

    axs[1, 2].scatter(skew_df_dep, kurt_df_dep, label="D", s=5, color=config.colors.dtu_red)
    axs[1, 2].scatter(skew_df_hel, kurt_df_hel, label="H", s=5, color=config.colors.bright_green)
    axs[1, 2].legend(loc="upper right")
    axs[1, 2].set_title("Skewness vs. kurtosis")

    plt.suptitle("Eyes open pre-treatment data features", fontsize="x-large")

    plt.show()

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
    """
