import os
import pickle
import time
import warnings
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yasa
from neurodsp.spectral import compute_spectrum
from omegaconf import OmegaConf
from scipy.signal import hilbert
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


def make_epochs(df, filename, seconds):
    # Input df: Table with all 2min for pre/post and EO/EC
    # No overlap
    subjects = df["Subject_ID"].unique()
    step = seconds
    end = int((56960 // samp_freq // seconds) * seconds + step)
    cuts = np.arange(0, end * samp_freq, step * samp_freq)
    df_10s_epochs = pd.DataFrame(columns=df.columns)
    df_10s_epochs["Epoch"] = None

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
            df_10s_epochs = df_10s_epochs.append(current_epoch)
            epoch += 1
        s += 1

    # Save table as .pickle file
    with open(root + "Epochs/10_seconds/" + filename, "wb") as f:
        pickle.dump(df_10s_epochs, f)

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


def plot_BIMFs(orig_sig, one_bimf, channel_name, h, K=5):
    # Visualize signal and it's decomposed modes
    fig = plt.figure(figsize=(10, 8))

    # Set up x-axis in time domain
    points = orig_sig.shape[0]
    x = np.linspace(0, points / samp_freq, points)

    # Plot
    plt.subplot(K + 1, 1, 1)
    plt.plot(x, orig_sig, color=color_codes[0], linewidth=0.75)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(channel_name + " signal")
    plt.title(f"Signal (red) and BIMFs (orange) - {h} subject")
    for m in range(K):
        plt.subplot(K + 1, 1, m + 2)
        plt.plot(x, one_bimf.iloc[:, m], color=color_codes[6], linewidth=0.75)
        plt.ylabel(f"BIMF{m+1}")
        plt.yticks([])
        if m == K - 1:
            plt.xticks(np.arange(11))
            plt.xlabel("Time [s]")
        else:
            plt.xticks([])
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def plot_BIMFs_h_and_d_examples(df10, all_bimfs, channel_name, h_sub, d_sub, epo):
    # VMD plot example
    example_bimfs = [
        "BIMF1-" + channel_name,
        "BIMF2-" + channel_name,
        "BIMF3-" + channel_name,
        "BIMF4-" + channel_name,
        "BIMF5-" + channel_name,
    ]

    # Original signals
    orig_sig_h = df10[(df10["Subject_ID"] == h_sub) & (df10["Epoch"] == epo)]
    orig_sig_h = orig_sig_h[channel_name]
    orig_sig_d = df10[(df10["Subject_ID"] == d_sub) & (df10["Epoch"] == epo)]
    orig_sig_d = orig_sig_d[channel_name]

    # Corresponding BIMFs
    one_bimf_set_h = all_bimfs[
        (all_bimfs["Subject_ID"] == h_sub) & (all_bimfs["Epoch"] == epo)
    ]
    one_bimf_set_h = one_bimf_set_h[example_bimfs]
    one_bimf_set_d = all_bimfs[
        (all_bimfs["Subject_ID"] == d_sub) & (all_bimfs["Epoch"] == epo)
    ]
    one_bimf_set_d = one_bimf_set_d[example_bimfs]

    # Plot
    fig1 = plot_BIMFs(orig_sig_h, one_bimf_set_h, channel_name, "healthy")
    fig2 = plot_BIMFs(orig_sig_d, one_bimf_set_d, channel_name, "depressed")

    return fig1, fig2


def analytic_BIMF(bimf):
    # Analytic representation of a BIMF (complex nubmers)
    h_transform = hilbert(bimf)
    a = bimf + h_transform
    return a


def stat_features_from_VMD(a_bimf):
    a_bimf = a_bimf.to_numpy()
    a_bimf = a_bimf.astype(np.complex128)

    # freqs: Frequencies at which the measure was calculated
    # spectrum: Power spectral density
    freqs, spectrum = compute_spectrum(sig=a_bimf, fs=samp_freq)

    # Spectral centroid
    cent = (sum(freqs * spectrum)) / sum(spectrum)

    # Spectral variance
    var = (sum((freqs - cent) ** 2 * spectrum)) / sum(spectrum)

    # Spectral skewness
    skew = (sum(((freqs - cent) / var) ** 3 * spectrum)) / sum(spectrum)

    return cent, var, skew


if __name__ == "__main__":
    # ---- VARIABLES ----
    current_data_file = "all_pre_EC_2min"
    # VMD
    run_vmd_section = True
    plot_vmd_examples = False
    save_vmd_features = True
    # Non-VMD
    run_non_vmd_section = False
    save_non_vmd_features = False

    # Load the data from a pickle file
    with open(root + "Epochs/Whole_rec/" + current_data_file + ".pickle", "rb") as f:
        current_dataset = pickle.load(f)

    # ------------------------------------------------ FEATURES SETUP ------------------------------------------------

    # Create all column names for feature matrices
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

    # Absolute spectral power, total and per brain region
    total_pow_names = ["Total-pow-" + chan for chan in channel_names]
    regional_pow_names = []
    for r in region_names:
        for w in wave_names:
            name = w + "-pow-" + r
            regional_pow_names.append(name)
        tot_name = "Total-pow-" + r
        regional_pow_names.append(tot_name)
    power_names = power_names + total_pow_names + regional_pow_names

    # All non-VMD feature names in one vector
    feature_names = moment_names + power_names

    # Features from BIMFs from VMD
    bimf_names = ["BIMF1", "BIMF2", "BIMF3", "BIMF4", "BIMF5"]
    bimf_features = ["C", "V", "S"]
    vmd_feature_names = []
    for f in bimf_features:
        for c in channel_names:
            for b in bimf_names:
                vmd_feature_names.append(b + "-" + f + "-" + c)

    # ------------------------------------- VMD FEATURE CALCULATIONS --------------------------------------

    if run_vmd_section:
        # Parameters for VMD proposed in epilepsy paper
        K = 5  # Number of decomposed nodes
        alpha = 9800  # Data-fidelity constraint parameter
        tau = 0  # Time step of dual ascent
        DC = 0  # Number of DC components
        init = 1  # Value of initial frequency for the decomposed modes
        tol = 1e-6  # Tolerance value for convergence criteria

        # Split data into 10-second epochs
        # Check whether we have already made and saved the combined data files
        check_file = (
            root + "Epochs/10_seconds/" + current_data_file[:-4] + "10s_epochs.pickle"
        )
        if not os.path.exists(check_file):
            make_epochs(
                current_dataset, current_data_file[:-4] + "10s_epochs.pickle", 10
            )

        # Run VMD and get BIMFs
        # Check whether we have already run and saved the VMD
        filename_split = current_data_file[:-4].split("_")
        pe = filename_split[1] + "_" + filename_split[2]
        check_file = root + "Epochs/10_seconds/BIMFs/all_BIMFs_" + pe + ".pickle"
        if not os.path.exists(check_file):
            # Get 10-second epoch data
            with open(
                root
                + "Epochs/10_seconds/"
                + current_data_file[:-4]
                + "10s_epochs.pickle",
                "rb",
            ) as f:
                df10 = pickle.load(f)

            # Remove reference electrodes from column names (just to shorten)
            df10.columns = df10.columns.str.replace("-A1", "")
            df10.columns = df10.columns.str.replace("-A2", "")

            # Fetch IDs
            subjects = df10["Subject_ID"].unique()
            epochs = df10["Epoch"].unique()
            channels = df10.columns[:-3]
            print("Calculating BIMFs")
            calculate_BIMFs(
                df10, subjects, channels, epochs, alpha, tau, K, DC, init, tol
            )

        # Get BIMFs
        with open(
            root + "Epochs/10_seconds/BIMFs/all_BIMFs_" + pe + ".pickle", "rb"
        ) as f:
            all_bimfs = pickle.load(f)

        # VMD plot examples
        if plot_vmd_examples:
            # Get 10-second epoch data
            with open(
                root
                + "Epochs/10_seconds/"
                + current_data_file[:-4]
                + "10s_epochs.pickle",
                "rb",
            ) as f:
                df10 = pickle.load(f)

            # Remove reference electrodes from column names (just to shorten)
            df10.columns = df10.columns.str.replace("-A1", "")
            df10.columns = df10.columns.str.replace("-A2", "")

            # Plot
            fig1, fig2 = plot_BIMFs_h_and_d_examples(df10, all_bimfs, "Oz", 12, 58, 0)
            plt.show()

        # Create empty lists for feature matrix and target vector
        vmd_feature_mat = []
        vmd_targets = []

        # Lists of subject and epoch numbers present in data table
        subject_ids = all_bimfs["Subject_ID"].unique()
        subject_ids = [int(x) for x in subject_ids]
        epochs = all_bimfs["Epoch"].unique()
        epochs = [int(x) for x in epochs]

        # Iterate over all subjects and get the BIMF features for each channel, for each subject
        # We calculate the features for each epoch, and then average over the epochs
        # Spectral centroid C_sp, spectral variance σ2_sp and spectral skewness β_sp
        for sub in subject_ids:
            print("Calculating BIMF features for subject", sub)
            sub_df = all_bimfs[all_bimfs["Subject_ID"] == sub]
            is_depressed = sub_df["Depressed"].iloc[0]
            epoch_mat = []
            for e in epochs:
                e_df = sub_df[sub_df["Epoch"] == e]
                e_df = e_df.iloc[:, :-3]
                sp_cents, sp_vars, sp_skews = [], [], []
                for col in e_df:
                    a = analytic_BIMF(e_df[col])
                    c, v, s = stat_features_from_VMD(a)
                    sp_cents.append(c)
                    sp_vars.append(v)
                    sp_skews.append(s)
                epoch_row = np.concatenate((sp_cents, sp_vars, sp_skews))
                epoch_mat.append(epoch_row)
            epoch_mat = np.array(epoch_mat)

            # We have 11 epochs, average over them for 1 number per channel, per subject
            epoch_mat_means = epoch_mat.mean(axis=0)

            # Inserting into the target vector and feature matrix
            vmd_targets.append([sub, is_depressed])
            vmd_feature_mat.append(epoch_mat_means)

        # ------------------------------------------- VMD FEATURE MATRIX -------------------------------------------

        # When feature matrix has been filled with values, we normalize it
        vmd_feature_mat = np.array(vmd_feature_mat).astype(float)
        vmd_feature_mat = zscore(vmd_feature_mat, axis=None)
        # vmd_feature_mat = zscore(vmd_feature_mat, axis=1)
        vmd_targets = np.array(vmd_targets)

        # Then put it into a dataframe so we have the column names
        feature_df = pd.DataFrame(vmd_feature_mat, columns=vmd_feature_names)
        feature_df["Subject_ID"] = vmd_targets[:, 0]
        feature_df["Depression"] = vmd_targets[:, 1]

        if save_vmd_features:
            # Save feature matrix as .pickle file
            with open(
                root + "Features/NEW-vmd_feature_df_" + current_data_file + ".pickle",
                "wb",
            ) as f:
                pickle.dump(feature_df, f)
            print("VMD feature matrix saved.")

    # ------------------------------------------- NON-VMD FEATURE CALCULATIONS -------------------------------------

    if run_non_vmd_section:
        # Create empty lists for feature matrix and target vector
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
            print("Calculating features for subject", sub)
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

        # ------------------------------------------------ NON-VMD FEATURE MATRIX ------------------------------------------------

        # When feature matrix has been filled with values, we normalize it
        feature_mat = np.array(feature_mat).astype(float)
        feature_mat = zscore(feature_mat, axis=None)
        # feature_mat = zscore(feature_mat, axis=1)
        targets = np.array(targets)

        # Then put it into a dataframe so we have the column names
        feature_df = pd.DataFrame(feature_mat, columns=feature_names)
        feature_df["Subject_ID"] = targets[:, 0]
        feature_df["Depression"] = targets[:, 1]

        if save_non_vmd_features:
            # Save feature matrix as .pickle file
            with open(
                root + "Features/NEW-feature_df_" + current_data_file + ".pickle",
                "wb",
            ) as f:
                pickle.dump(feature_df, f)
            print("Non-VMD feature matrix saved.")
