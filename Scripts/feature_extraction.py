import os
import pickle
# import random
import time
import warnings
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yasa
# from dit.other import renyi_entropy
from omegaconf import OmegaConf
from scipy.stats import kurtosis, skew, zscore
from vmdpy import VMD

warnings.filterwarnings("ignore")


# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
samp_freq = config.sample_frequency
healthy_num = config.subject_classes.healthy_num
depressed_num = config.subject_classes.depressed_num
all_electrodes = config.electrodes.all_electrodes_arranged
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


def make_epochs_overlap(df, seconds, overlap):
    # Input df: Table with all 2min for pre/post and EO/EC
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
    with open(
        root + "Epochs/10_seconds" + "/all_pre_EC_10s_epochs_overlap.pickle", "wb"
    ) as f:
        pickle.dump(all_pre_EC_10s_epochs, f)

    return


def make_epochs(df, seconds):
    # Input df: Table with all 2min for pre/post and EO/EC
    subjects = df["Subject_ID"].unique()
    step = seconds
    end = int((56960 // samp_freq // seconds) * seconds + step)
    cuts = np.arange(0, end * samp_freq, step * samp_freq)
    all_pre_EC_10s_epochs = pd.DataFrame(columns=df.columns)
    all_pre_EC_10s_epochs["Epoch"] = None

    s = 1
    for subject in subjects:
        current_sub = df[df["Subject_ID"] == subject]
        epoch = 0
        print(f"Making epochs for subject {s}/{subjects.shape[0]}")
        for i in range(len(cuts[:-1])):
            start = cuts[i]
            end = cuts[i + 1]
            current_epoch = current_sub.iloc[start:end, :]
            current_epoch["Epoch"] = epoch
            all_pre_EC_10s_epochs = all_pre_EC_10s_epochs.append(current_epoch)
            epoch += 1
        s += 1

    # Save table as .pickle file
    with open(root + "Epochs/10_seconds" + "/all_pre_EC_10s_epochs.pickle", "wb") as f:
        pickle.dump(all_pre_EC_10s_epochs, f)

    return


def calculate_BIMFs(df, subjects, channels, epochs, alpha, tau, K, DC, init, tol):
    # Calculates BIMFs of each channel, per subject, per epoch
    # Saves dataframe for each channel (with all BIMFs for all subjects and their epochs)
    bimf_cols = ["BIMF1-", "BIMF2-", "BIMF3-", "BIMF4-", "BIMF5-"]

    # Outputs from VMD function:
    # u       - the collection of decomposed modes/BIMFs
    # u_hat   - spectra of the modes
    # omega   - estimated mode center-frequencies

    # Start loop
    i = 1
    start_time = time.time()
    for c in channels:
        print(f"--------------- CHANNEL {c} ({i}/{channels.shape[0]}) ---------------")
        chan_df = df[[c, "Subject_ID", "Epoch"]]
        bimf_df_cols = [b + c for b in bimf_cols]
        cols = bimf_df_cols + ["Subject_ID", "Epoch"]
        bimf_df = pd.DataFrame(columns=cols)
        for s in subjects:
            print("  Subject", s)
            sub_df = chan_df[chan_df["Subject_ID"] == s]
            for e in epochs:
                print("    Epoch ", e)
                epoch_df = sub_df[sub_df["Epoch"] == e]
                f = epoch_df[c]
                u, _, _ = VMD(f, alpha, tau, K, DC, init, tol)
                su = np.ones(u.shape[1]) * s
                ep = np.ones(u.shape[1]) * e
                app = u.T
                app = np.c_[app, su]
                app = np.c_[app, ep]
                bimf_df = bimf_df.append(
                    pd.DataFrame(app, columns=cols), ignore_index=True
                )
        # Save as .pickle
        path = (
            root + "Epochs/10_seconds/BIMFs/" + current_data_file[:-4] + c + ".pickle"
        )
        with open(path, "wb") as f:
            pickle.dump(bimf_df, f)
        print(f"BIMFs of channel {c} saved!")
        current_time = time.time()
        print("Timestamp:", current_time - start_time, "seconds")
        print()
        i += 1

    return


def spectral_centroid():
    return


if __name__ == "__main__":
    # Load the data from a pickle file
    current_data_file = "all_pre_EC_2min"
    with open(root + "Epochs/Whole_rec/" + current_data_file + ".pickle", "rb") as f:
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

    # All feature names (not from VMD) in one vector
    feature_names = moment_names + power_names

    # Features from BIMFs
    AM_names = ["AM-" + chan for chan in channel_names]
    FM_names = ["FM-" + chan for chan in channel_names]
    C_names = ["C-" + chan for chan in channel_names]
    sigma2_names = ["sigma2-" + chan for chan in channel_names]
    beta_names = ["beta-" + chan for chan in channel_names]

    # Features from BIMFs table names
    BIMF_feature_names = (
        AM_names
        + FM_names
        + C_names
        + sigma2_names
        + beta_names
        + ["Subject_ID", "Depressed", "Epoch", "BIMF"]
    )

    # ------------------------------------------------ VMD ------------------------------------------------

    # Parameters for VMD proposed in epilepsy paper
    K = 5  # Number of decomposed nodes
    alpha = 9800  # Data-fidelity constraint parameter
    tau = 0  # Time step of dual ascent
    DC = 0  # Number of DC components
    init = 1  # Value of initial frequency for the decomposed modes
    tol = 1e-6  # Tolerance value for convergence criteria

    # Split data into 10-second epochs
    # Check whether we have already made and saved the combined data files
    check_file = root + "Epochs/10_seconds/all_pre_EC_10s_epochs.pickle"
    if not os.path.exists(check_file):
        make_epochs(current_dataset, 10)
    with open(root + "Epochs/10_seconds" + "/all_pre_EC_10s_epochs.pickle", "rb") as f:
        df10 = pickle.load(f)

    # Remove reference electrodes from column names (just to shorten)
    df10.columns = df10.columns.str.replace("-A1", "")
    df10.columns = df10.columns.str.replace("-A2", "")

    # Fetch IDs
    subjects = df10["Subject_ID"].unique()
    epochs = df10["Epoch"].unique()
    channels = df10.columns[:-3]

    # Run VMD and get BIMFs
    # Check whether we have already run and saved the VMD
    check_file = (
        root + "Epochs/10_seconds/BIMFs/" + current_data_file[:-4] + "Fp1.pickle"
    )
    if not os.path.exists(check_file):
        print("Calculating BIMFs")
        calculate_BIMFs(df10, subjects, channels, epochs, alpha, tau, K, DC, init, tol)

    # Get BIMFs
    with open(root + "Epochs/10_seconds/BIMFs/all_pre_EC_C3.pickle", "rb") as f:
        bimfs_fp1 = pickle.load(f)

    # Analytic representation of BIMFs
    # Amplitude modulation bandwidth AM_Bω
    # Frequency modulation bandwidth FM_Bω
    # Spectral centroid C_sp
    # Spectral variance σ2_sp
    # Spectral skewness β_sp

    # VMD plot example
    ex_sub = 60
    ex_epo = 5
    orig_sig = df10[(df10["Subject_ID"] == ex_sub) & (df10["Epoch"] == ex_epo)]
    orig_sig = orig_sig["C3"]
    one_bimf = bimfs_fp1[
        (bimfs_fp1["Subject_ID"] == ex_sub) & (bimfs_fp1["Epoch"] == ex_epo)
    ]
    one_bimf = one_bimf[one_bimf.columns[0:5]]

    # Set up x-axis in time domain
    points = one_bimf.shape[0]
    x = np.linspace(0, points / samp_freq, points)

    # Visualize decomposed modes
    plt.figure(figsize=(8, 8))
    plt.subplot(K + 1, 1, 1)
    plt.plot(x, orig_sig, color=color_codes[0])
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("C3 signal")
    plt.title("Signal (red) and BIMFs (orange)")
    for m in range(K):
        plt.subplot(K + 1, 1, m + 2)
        plt.plot(x, one_bimf.iloc[:, m], color=color_codes[6])
        plt.ylabel(f"BIMF{m+1}")
        plt.yticks([])
        if m == K - 1:
            plt.xticks(np.arange(11))
            plt.xlabel("Time [s]")
        else:
            plt.xticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    # ------------------------------------------- NON-VMD FEATURE CALCULATIONS -------------------------------------

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
    sub_df10 = df10[(df10["Subject_ID"] == 5) & (df10["Epoch"] == 5)]
    f = sub_df10["Fp1-A1"].to_numpy()

    # Find K using Renyi's entropy criterion from Sada's paper
    # renyi = renyi_entropy(f, 2, rvs=None, rv_mode=None)
    # print(renyi)

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

    # Step 1: Parameters for VMD proposed in the paper
    K = np.arange(1, 9, 1)  # Number of decomposed nodes
    alpha = np.arange(100, 10000, 100)  # Data-fidelity constraint parameter
    tau = 0  # Time step of dual ascent
    DC = 0  # Number of DC components
    init = 1  # Value of initial frequency for the decomposed modes
    tol = 1e-6  # Tolerance value for convergence criteria

    # Step 2: Split data into 10-second epochs
    # make_epochs(all_pre_EC_116s, 10, 0.5)  # --> Already done, comment out
    with open(root + "Epochs/10_seconds" + "/all_pre_EC_10s_epochs.pickle", "rb") as f:
        df10 = pickle.load(f)

    subjects = df10["Subject_ID"].unique()
    epochs = df10["Epoch"].unique()
    channels = df10.columns[:-3]

    # Step 3: Run VMD
    # Outputs:
    # u       - the collection of decomposed modes/BIMFs
    # u_hat   - spectra of the modes
    # omega   - estimated mode center-frequencies

    # Track for healthy and depressed separately
    max_kurt_h = 0
    best_K_h = 0
    best_alpha_h = 0
    max_kurt_d = 0
    best_K_d = 0
    best_alpha_d = 0
    i = 1

    start_time = time.time()
    for s in subjects:
        c = random.sample(sorted(channels), 1)[0]
        e = int(random.sample(sorted(epochs), 1)[0])
        print(f"Subject {i}/{subjects.shape[0]}")
        print("  Epoch", e)
        print("    Channel", c)
        sub_df = df10[(df10["Subject_ID"] == s) & (df10["Epoch"] == e)]
        isDep = sub_df["Depressed"].max()
        f = sub_df[c]
        for k in K:
            print("      K =", k)
            for a in alpha:
                u, _, _ = VMD(f, a, tau, k, DC, init, tol)
                u_sum = u.sum(axis=0)
                kurt = kurtosis(u_sum)
                if (kurt > max_kurt_h) and (isDep == 0):
                    max_kurt_h = kurt
                    best_K_h = k
                    best_alpha_h = a
                elif (kurt > max_kurt_d) and (isDep == 1):
                    max_kurt_d = kurt
                    best_K_d = k
                    best_alpha_d = a
        print(f"Best healthy: {max_kurt_h}, {best_K_h}, {best_alpha_h}")
        print(f"Best depressed: {max_kurt_d}, {best_K_d}, {best_alpha_d}")
        current_time = time.time()
        print("Timestamp:", current_time - start_time)
        print()
        i += 1

    print("Maximum kurtosis obtained (h):", max_kurt_h)
    print("Best K (h):", best_K_h)
    print("Best alpha (h):", best_alpha_h)

    print("Maximum kurtosis obtained (d):", max_kurt_d)
    print("Best K (d):", best_K_d)
    print("Best alpha (d):", best_alpha_d)

    start_time = time.time()
    for s in subjects:
        random_electrodes = random.sample(sorted(channels), 2)
        random_epochs = random.sample(sorted(epochs), 2)
        print(f"Subject {i}/{subjects.shape[0]}")
        for e in random_epochs:
            print("  Epoch", e)
            sub_df = df10[(df10["Subject_ID"] == s) & (df10["Epoch"] == e)]
            f = sub_df[sub_df.columns.intersection(random_electrodes)]
            isDep = sub_df["Depressed"].iloc[0]
            for c in random_electrodes:
                f = sub_df[c]
                print("    Channel", c)
                for k in K:
                    print("      K =", k)
                    for a in alpha:
                        u, _, _ = VMD(f, a, tau, k, DC, init, tol)
                        u_sum = u.sum(axis=0)
                        kurt = kurtosis(u_sum)
                        if (kurt > max_kurt_h) and (isDep == 0):
                            max_kurt_h = kurt
                            best_K_h = k
                            best_alpha_h = a
                        elif (kurt > max_kurt_d) and (isDep == 1):
                            max_kurt_d = kurt
                            best_K_d = k
                            best_alpha_d = a
        print(f"Best healthy: {max_kurt_h}, {best_K_h}, {best_alpha_h}")
        print(f"Best depressed: {max_kurt_d}, {best_K_d}, {best_alpha_d}")
        current_time = time.time()
        print("Timestamp:", current_time - start_time)
        print()
        i += 1

    print("Maximum kurtosis obtained (h):", max_kurt_h)
    print("Best K (h):", best_K_h)
    print("Best alpha (h):", best_alpha_h)

    print("Maximum kurtosis obtained (d):", max_kurt_d)
    print("Best K (d):", best_K_d)
    print("Best alpha (d):", best_alpha_d)

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

    plt.show() """
