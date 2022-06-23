import pickle
import warnings
from datetime import datetime
from sys import platform

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yasa
from omegaconf import OmegaConf
from scipy.stats import kurtosis, skew, zscore
from vmdpy import VMD

warnings.filterwarnings("ignore")


# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
samp_freq = config.sample_frequency
healthy_num = config.subject_classes.healthy_num
depressed_num = config.subject_classes.depressed_num

# Get root folder based on which operating system I'm working on
root = "Data/tDCS_EEG_data/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root


# def get_subject_id(filename):
#     name_split = filename.split("_")
#     subject = name_split[0][1:]
#     id = int(subject)
#     return id


def make_epochs(df, seconds, overlap):
    # Input df: Table with all 116s for pre/post and EO/EC
    subjects = df["Subject_ID"].unique()
    step = int(seconds * overlap)
    end = int(
        (116 // seconds) * seconds + step
    )  # Fix/check if I make different epochs than 10s!!
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
    with open(root + "Epochs/10_seconds" + "/all_pre_EC_10s_epochs.pickle", "wb") as f:
        pickle.dump(all_pre_EC_10s_epochs, f)

    return


if __name__ == "__main__":
    # Let's start by looking only at EYES CLOSED AND PRE DATA, since we are working on task 1 (dep or healthy)
    # Load the data from a pickle file
    with open(root + "Epochs/116_seconds" + "/all_pre_EC_116s.pickle", "rb") as f:
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

    # ---------------------------------------------------------------------------------------
    # Band power trial!
    # From:
    # https://raphaelvallat.com/bandpower.html
    # https://raphaelvallat.com/yasa/build/html/generated/yasa.bandpower.html#yasa.bandpower
    # https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb

    current_sub = all_pre_EC_116s[all_pre_EC_116s["Subject_ID"] == 12]
    current_sub = current_sub.iloc[:, :-2]
    current_sub = current_sub.to_numpy()

    bp = yasa.bandpower(
        current_sub.T, sf=samp_freq, ch_names=channel_names, relative=False
    )
    print(bp)

    # ---------------------------------------------------------------------------------------

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
        current_sub = current_sub.iloc[:, :-2]
        current_sub = current_sub.to_numpy()

        # Calculate statistics per channel
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

    save_features = True
    if save_features:
        # Datetime object containing current date and time, for saving feature_df files
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%y_%H:%M:%S")

        # Save feature matrix as .pickle file
        with open(root + "Features_and_output" + "/feature_df.pickle", "wb") as f:
            pickle.dump(feature_df, f)

        print("Feature matrix saved.")

    # ------------------------------------------------ VMD STUFF ------------------------------------------------

    """
    PAPER: Epilepsy seizure detection using kurtosis based VMD's parameters selection and bandwidth features
    AUTHORS: Sukriti, Monisha Chakraborty, Debjani Mitra

    The proposed methodology is described as follows:

    Step 1: A range of K and alpha is set such that K = 1-15 with an interval of 1 and
            alpha = 100-15000 with an interval of 100.
    Step 2: A new signal is formulated such that it consists of 10 s of EEG segments from each of the datasets
            Z, F and S.
    Step 3: The new signal is decomposed by VMD for all possible combinations of K and alpha to obtain K BIMFs.
    Step 4: The BIMFs are then summed up to obtain the reconstruction signal for each decomposition.
    Step 5: Kurtosis of each reconstructed signal is determined.
    Step 6: The value of K and alpha for which kurtosis is maximum is recorded.
    Step 7: Lastly, the five sets of EEG data (Z, O, N, F and S) are decomposed into BIMFs by VMD under the
            recorded value of K and alpha.
    """

    # Step 1: Parameters for VMD proposed in the paper
    K = np.arange(1, 16, 1)  # Number of decomposed nodes
    alpha = np.arange(100, 15100, 100)  # Data-fidelity constraint parameter
    tau = 0  # Time step of dual ascent
    DC = 0  # Number of DC components
    init = 1  # Value of initial frequency for the decomposed modes
    tol = 1e-6  # Tolerance value for convergence criteria

    # Step 2: Split data into 10-second epochs
    # make_epochs(all_pre_EC_116s, 10, 0.5)  # --> Already done, comment out
    with open(root + "Epochs/10_seconds" + "/all_pre_EC_10s_epochs.pickle", "rb") as f:
        df10 = pickle.load(f)

    # Step 3: Run VMD
    # Outputs:
    # u       - the collection of decomposed modes/BIMFs
    # u_hat   - spectra of the modes
    # omega   - estimated mode center-frequencies

    subjects = df10["Subject_ID"].unique()
    epochs = df10["Epoch"].unique()
    channels = df10.columns[:-3]

    """ sub_df10 = df10[(df10["Subject_ID"] == 12) & (df10["Epoch"] == 0)]
    f = sub_df10["Fp1-A1"]

    K = 3
    alpha = 2000

    u, _, _ = VMD(f, alpha, tau, K, DC, init, tol)
    u_sum = u.sum(axis=0)

    #. Visualize decomposed modes
    plt.figure(figsize=(14,14))
    plt.subplot(5,1,1)
    plt.plot(f)
    plt.title('Original signal')
    plt.subplot(5,1,2)
    plt.plot(u[0].T)
    plt.title('Decomposed mode 1')
    plt.subplot(5,1,3)
    plt.plot(u[1].T)
    plt.title('Decomposed mode 2')
    plt.subplot(5,1,4)
    plt.plot(u[2].T)
    plt.title('Decomposed mode 3')
    plt.subplot(5,1,5)
    plt.plot(u_sum.T)
    plt.title('Reconstructed signal')
    plt.xlabel('Datapoints')
    #plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
    #plt.tight_layout()
    plt.show() """

    max_kurt = 0
    best_K = 0
    best_alpha = 0
    i = 1

    for s in subjects:
        print(f"Subject {i}/{subjects.shape[0]}")
        for e in epochs:
            print("  Epoch ", e)
            sub_df = df10[(df10["Subject_ID"] == s) & (df10["Epoch"] == e)]
            f = sub_df["Fp1-A1"]
            # for c in channels:
            #     sub_df = df10[(df10["Subject_ID"] == s) & (df10["Epoch"] == e)]
            #     f = sub_df[c]
            #     print("    Channel ", c)
            for k in K:
                for a in alpha:
                    u, _, _ = VMD(f, a, tau, k, DC, init, tol)
                    u_sum = u.sum(axis=0)
                    kurt = kurtosis(u_sum)
                    if kurt > max_kurt:
                        max_kurt = kurt
                        best_K = k
                        best_alpha = a
        i += 1

    print("Maximum kurtosis obtained:", max_kurt)
    print("Best K:", best_K)
    print("Best alpha:", best_alpha)

    """
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
