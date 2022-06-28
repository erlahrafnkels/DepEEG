import pickle
import warnings
from sys import platform

import matplotlib.pyplot as plt
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
frontal = config.electrodes.frontal
temporal = config.electrodes.temporal
parietal = config.electrodes.parietal
occipital = config.electrodes.occipital
central = config.electrodes.central
color_codes = [c[1] for c in config.colors.items()]

# Get root folder based on which operating system I'm working on
root = "Data/tDCS_EEG_data/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root


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
    # Load the data from a pickle file
    current_data_file = "all_pre_EC_116s"
    with open(root + "Epochs/116_seconds/" + current_data_file + ".pickle", "rb") as f:
        current_dataset = pickle.load(f)

    # ------------------------------------------------ FEATURES SETUP ------------------------------------------------

    # Create all column names for feature matrix
    channel_names = []
    if any(x in current_data_file for x in ["EC", "EO"]):
        channel_names = current_dataset.columns[:-2]
    else:
        channel_names = current_dataset.columns[:-3]
    channel_names = [c[:-3] for c in channel_names]
    wave_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    region_names = ["frontal", "temporal", "parietal", "occipital", "central"]
    regions = [frontal, temporal, parietal, occipital, central]

    # The first four moments of distribution: Mean, variance, skewness and kurtosis
    mean_names = ["Mean-" + chan for chan in channel_names]
    var_names = ["Var-" + chan for chan in channel_names]
    skew_names = ["Skew-" + chan for chan in channel_names]
    kurt_names = ["Kurt-" + chan for chan in channel_names]
    moment_names = mean_names + var_names + skew_names + kurt_names

    # Absolute spectral power of each frequency band
    delta_pow_names = ["Delta-pow-" + chan for chan in channel_names]
    theta_pow_names = ["Theta-pow-" + chan for chan in channel_names]
    alpha_pow_names = ["Alpha-pow-" + chan for chan in channel_names]
    beta_pow_names = ["Beta-pow-" + chan for chan in channel_names]
    gamma_pow_names = ["Gamma-pow-" + chan for chan in channel_names]
    power_names = (
        delta_pow_names
        + theta_pow_names
        + alpha_pow_names
        + beta_pow_names
        + gamma_pow_names
    )

    total_pow_names = ["Total-pow-" + chan for chan in channel_names]
    regional_pow_names = []
    for r in region_names:
        for w in wave_names:
            name = w + "-pow-" + r
            regional_pow_names.append(name)
        tot_name = "Total-pow-" + r
        regional_pow_names.append(tot_name)
    power_names = power_names + total_pow_names + regional_pow_names

    # All feature names in one vector
    feature_names = moment_names + power_names

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

    sub_df10 = df10[(df10["Subject_ID"] == 5) & (df10["Epoch"] == 0)]
    f = sub_df10["Fp1-A1"]

    K = 9
    alpha = 9800

    u, _, _ = VMD(f, alpha, tau, K, DC, init, tol)
    u_sum = u.sum(axis=0)
    modes = len(u)

    # Visualize decomposed modes
    plt.figure(figsize=(8, 8))
    plt.subplot(K + 1, 1, 1)
    plt.plot(f, color=color_codes[0])
    plt.xticks([])
    plt.yticks([])
    plt.title("Original signal (red) and BIMFs (orange)")
    for m in range(K):
        plt.subplot(K + 1, 1, m + 2)
        plt.plot(u[m].T, color=color_codes[6])
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    # ------------------------------------------------ FEATURE CALCULATIONS ------------------------------------------

    # Create empty lists for feature matrix
    feature_mat = []
    targets = []

    # Specify frequency bands for spectral power calculations
    bands = [
        (0.5, 4, "Delta"),
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (12, 35, "Beta"),
        (35, 40, "Gamma"),
    ]

    # List of subject numbers present in data table
    subject_ids = current_dataset["Subject_ID"].unique()

    # Iterate over all subjects and get the features for each channel, for each subject
    for sub in subject_ids:
        # Get subtable for subject and depression value
        current_sub = current_dataset[current_dataset["Subject_ID"] == sub]
        is_depressed = current_sub["Depressed"].iloc[0]
        if any(x in current_data_file for x in ["EC", "EO"]):
            current_sub = current_sub.iloc[:, :-2]
        else:
            current_sub = current_sub.iloc[:, :-3]
        current_sub = current_sub.to_numpy()

        # Calculate moment statistics per channel
        means = np.mean(current_sub, axis=0)
        vars = np.var(current_sub, axis=0)
        skews = skew(current_sub, axis=0)
        kurts = kurtosis(current_sub, axis=0)

        # Calculate absolute spectral power of each frequency band, per channel
        # From https://raphaelvallat.com/yasa/build/html/generated/yasa.bandpower.html#yasa.bandpower
        bp = yasa.bandpower(
            current_sub.T,
            sf=samp_freq,
            ch_names=channel_names,
            bands=bands,
            relative=False,
        )
        bp = bp.T

        # Power of each wave per channel
        bp_power = []
        for i in range(6):
            bp_power.append(bp.iloc[i])

        # Mean power for each brain region
        regions_power = []
        for r in regions:
            for w in bp_power:
                pow = w.loc[r].to_numpy()
                regions_power.append(np.mean(pow))

        # Convert waves-per-channel vectors to numpy
        bp_power_np = []
        for w in bp_power:
            bp_power_np.append(w.to_numpy())

        # And flatten the list
        bp_power_np_flat = [x for xs in bp_power_np for x in xs]

        # Concatenate arrays to make the whole row for the subject, for inserting into the feature matrix
        feature_row = np.concatenate(
            (means, vars, skews, kurts, bp_power_np_flat, regions_power)
        )
        targets.append([sub, is_depressed])
        feature_mat.append(feature_row)

    # ------------------------------------------------ FEATURE MATRIX ------------------------------------------------

    # When feature matrix has been filled with values, we normalize it
    feature_mat = np.array(feature_mat).astype(float)
    feature_mat = zscore(feature_mat, axis=None)
    # feature_mat = zscore(feature_mat, axis=1)
    targets = np.array(targets)

    # Then put it into a dataframe so we have the column names
    feature_df = pd.DataFrame(feature_mat, columns=feature_names)
    feature_df["Subject_ID"] = targets[:, 0]
    feature_df["Depression"] = targets[:, 1]

    save_features = False
    if save_features:
        # Save feature matrix as .pickle file
        with open(
            root + "Features_and_output/feature_df_" + current_data_file + ".pickle",
            "wb",
        ) as f:
            pickle.dump(feature_df, f)
        print("Feature matrix saved.")

    """
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
    """
