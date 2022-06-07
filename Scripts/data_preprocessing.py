import math as m
import os
import re
import warnings
from sys import platform

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf
from plot_signals import plot_record
from plot_title import make_plot_title
from scipy import signal
from scipy.stats import zscore
from sklearn.decomposition import FastICA

warnings.filterwarnings("ignore")


# Get global variables from the configuration file
config = OmegaConf.load("config.yaml")

# EEG electrodes
# OBS: The T3 electrode is sometimes referred to A2 instead of A1
# We don't let it bother us for now as it likely doesn't have much effect on calculations
electrodes_arranged = config.electrodes.all_electrodes_arranged
els_with_t3a1 = electrodes_arranged.copy()
els_with_t3a1.insert(13, "T3-A1")
els_with_t3a2 = electrodes_arranged.copy()
els_with_t3a2.insert(13, "T3-A2")
A1ref = config.electrodes.reference_electrodes.A1ref
A2ref = config.electrodes.reference_electrodes.A2ref
samp_freq = config.sample_frequency

# Subjects
new_data = config.new_data
healthy = config.subject_classes.healthy
depressed_active = config.subject_classes.depressed_active
depressed_sham = config.subject_classes.depressed_sham

# DTU colors for plots
color_codes = [c[1] for c in config.colors.items()]

# Get root folder based on which operating system I'm working on
root = ""
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/"


def get_files(data_path: str) -> list:
    subject_folders = os.listdir(data_path)
    txt_file_paths = []

    for subject in subject_folders:
        if subject == ".DS_Store":  # This file keeps creeping in, skip it
            continue
        path = data_path + subject
        subfolders = os.listdir(path)
        for p_folders in subfolders:
            str = path + "/" + p_folders
            pre_post = os.listdir(str)
            for file in pre_post:
                if file.endswith(".txt"):
                    file_path = str + "/" + file
                    txt_file_paths.append(file_path)

    return txt_file_paths


def update_filename(file: str) -> str:
    filename = os.path.basename(file)[:-4]
    subject = re.split("-|_", filename)[0]
    pre_or_post = ""
    open_or_closed = ""

    # Pre or post?
    if re.search("pre", filename, re.IGNORECASE):
        pre_or_post = "pre"
    elif re.search("post", filename, re.IGNORECASE):
        pre_or_post = "post"
    else:
        print(filename, " - missing pre or post")

    # Eyes open or closed?
    if re.search("EO", filename, re.IGNORECASE):
        open_or_closed = "EO"
    elif re.search("EC", filename, re.IGNORECASE):
        open_or_closed = "EC"
    else:
        print(filename, " - missing EO or EC")

    # Updated filename
    new_filename = subject + "_" + pre_or_post + "_" + open_or_closed + ".txt"

    return new_filename, subject


def correct_refs(df: pd.DataFrame) -> pd.DataFrame:
    df.update(df[A1ref].sub(df["A1"], axis=0))
    df.columns = [x + "-A1" if x in A1ref else x for x in df]
    df.update(df[A2ref].sub(df["A2"], axis=0))
    df.columns = [x + "-A2" if x in A2ref else x for x in df]
    return df


def clean_data(files: list):
    print("Cleaning data files.")
    for file in files:
        if ".txt" in file:
            df = pd.read_csv(file, sep="\t", index_col=False)
        else:
            data = mne.io.read_raw_eeglab(file)
            df = mne.io.Raw.to_data_frame(data)

        # If EEG is in column name, delete it
        df.columns = df.columns.str.replace("EEG ", "")

        # First, subtract A1 and A2 references from corresponding columns if it hasn't been done
        if not ("-A1" or "-A2") in df.columns[1]:
            df = correct_refs(df)

        # Then, remove all unnecessary columns and arrange correctly
        if "T3-A1" in df.columns:
            df = df[df.columns.intersection(els_with_t3a1)]
            df = df[els_with_t3a1]
        else:
            df = df[df.columns.intersection(els_with_t3a2)]
            df = df[els_with_t3a2]

        # Update filename and save
        filename, subject = update_filename(file)
        path = root + "Data/tDCS_EEG_data/Data_cleaned/" + subject + "/"
        if not os.path.isdir(path):
            os.makedirs(path)
        df.to_csv(path + filename, sep="\t", index=False)

    return


def notch_filter(
    data: pd.DataFrame, low_freq: float, high_freq: float
) -> list[float, float, np.ndarray]:
    notch_freq = 50  # Frequency to be removed from signal (Hz)
    quality_factor = (high_freq - low_freq) / m.sqrt(low_freq * high_freq)
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
    signal_notched = signal.filtfilt(b_notch, a_notch, data)
    return freq, h, signal_notched


def butter_bandpass_filter(
    data: np.ndarray, low_freq: float, high_freq: float, order: int = 5
) -> np.ndarray:
    nyq = 0.5 * samp_freq
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.filtfilt(b, a, data)
    return y


def filter_record(
    filename: str, subject: str, low_freq: float, high_freq: float
) -> pd.DataFrame:
    # Read in raw, cleaned data
    data_path = root + "Data/tDCS_EEG_data/Data_cleaned/" + subject + "/"
    data = pd.read_csv(data_path + filename, sep="\t", index_col=False)

    # For each channel apply filters, normalize and replace data column values with new
    for col in data.columns:
        channel = data[col]
        _, _, signal_notched = notch_filter(channel, low_freq, high_freq)
        signal_filtered = butter_bandpass_filter(signal_notched, low_freq, high_freq)
        sf_norm = zscore(signal_filtered)
        data[col] = sf_norm

    return data


def run_ICA(record: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame]:
    ica = FastICA(random_state=42)
    comps_ = ica.fit_transform(record)  # Reconstruct signals (type: np.ndarray)
    mixing_ = ica.mixing_  # Get estimated mixing matrix (type: np.ndarray)

    # Convert from numpy array to pandas dataframe
    comps = pd.DataFrame(comps_)
    mixing = pd.DataFrame(mixing_)

    return comps, mixing


def plot_example_process(
    filename: str, subject: str, low_freq: float, high_freq: float
) -> plt.figure:
    # Read in raw, cleaned data
    data_path = root + "Data/tDCS_EEG_data/Data_cleaned/" + subject + "/"
    data = pd.read_csv(data_path + filename, sep="\t", index_col=False)
    raw = data.iloc[:, 0]  # Here we're only looking at the records of Fp1

    # Filter out the 50 Hz powerline interference
    freq, h, signal_notched = notch_filter(raw, low_freq, high_freq)

    # Apply bandpass filter
    signal_filtered = butter_bandpass_filter(signal_notched, low_freq, high_freq)
    sf_norm = zscore(signal_filtered)

    # Set up x-axis in time domain
    points = raw.shape[0]
    x = np.linspace(0, points / samp_freq, points)

    # Make one big plot
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    # Raw signal
    ax1.plot(x, raw, color=config.colors.dtu_red)
    ax1.set_title("Raw signal")
    ax1.set_xlabel("Time [s]")
    ax1.grid()

    # Notch filter
    # In the plot we transform h from complex numbers to dBs
    ax2.plot(freq, 20 * np.log10(abs(h)), color=config.colors.dtu_red)
    ax2.set_title("Notch filter")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [dB]")
    ax2.grid()

    # Power spectral density of the raw, notched and filtered signals
    ax3.psd(raw, Fs=samp_freq, color=config.colors.dtu_red)
    ax3.psd(signal_notched, Fs=samp_freq, color=config.colors.black)
    ax3.psd(signal_filtered, Fs=samp_freq, color=config.colors.blue)
    ax3.legend(labels=["Raw", "Notched", "Bandpass + notched"])
    ax3.set_title("Power spectral density")

    # Original signal with the filters
    ax4.plot(x, signal_filtered, color=config.colors.dtu_red)
    ax4.set_title("Data filtered")
    ax4.set_xlabel("Time [s]")
    ax4.grid()

    # Original signal with the filters and normalized
    ax5.plot(x, sf_norm, color=config.colors.dtu_red)
    ax5.set_title("Data filtered and normalized")
    ax5.set_xlabel("Time [s]")
    ax5.grid()

    fig.suptitle(make_plot_title(filename), fontsize="xx-large")
    fig.tight_layout()

    return fig


def filter_and_ica(data_path: str, subjects: list):
    for sub in subjects:
        files = os.listdir(data_path + sub + "/")
        for file in files:
            rec = filter_record(file, sub, 0.5, 40)
            comps, mixing = run_ICA(rec)

            # Save all dataframes
            rec.to_csv(
                data_path + sub + "/" + file[:-4] + "_filt.txt", sep="\t", index=False
            )
            comps.to_csv(
                data_path + sub + "/" + file[:-4] + "_ICA_comps.txt",
                sep="\t",
                index=False,
            )
            mixing.to_csv(
                data_path + sub + "/" + file[:-4] + "_ICA_mix.txt",
                sep="\t",
                index=False,
            )

    return


def remove_artifacts(
    record: pd.DataFrame,
    ica_comps: pd.DataFrame,
    ica_mixing: pd.DataFrame,
    comp_ids: list,
) -> list[pd.DataFrame, pd.DataFrame]:
    # Make copies of dataframes to make sure they are not overwritten
    rec_copy = record.copy()
    comps_copy = ica_comps.copy()
    mix_copy = ica_mixing.copy()

    # Save column names and convert dataframes to numpy arrays for calculation
    columns = rec_copy.columns
    mix_np = mix_copy.to_numpy()
    comps_np = comps_copy.to_numpy()

    # Set all components with unwanted artefacts to zero
    for id in comp_ids:
        comps_np[:, id] = 0

    # Update record so unwanted artifacts are removed
    updated_rec = np.dot(comps_np, mix_np.T)
    updated_rec = pd.DataFrame(updated_rec, columns=columns)
    updated_comps = pd.DataFrame(comps_np)

    return updated_comps, updated_rec


def cut_record_116s(record: pd.DataFrame):
    # Cut off first 4 seconds and cut off after 120 seconds
    begin = 4 * samp_freq
    end = 120 * samp_freq

    # Update
    record.drop(record.index[0:begin], inplace=True)
    record.drop(record.index[end:], inplace=True)

    return record


if __name__ == "__main__":
    # Run this if we have don't have the cleaned files
    if not os.listdir(root + "Data/tDCS_EEG_data/Data_cleaned/"):
        files = get_files(root + "Data/tDCS_EEG_data/")
        clean_data(files)
        print("Cleaned files saved.")

    # Filtering and ICA
    # We only want to run this ONCE so we are always using the same ICA
    check_file = root + "Data/tDCS_EEG_data/Data_cleaned/S1/S1_pre_EO_ICA_mix.txt"
    if not os.path.exists(check_file):
        data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
        subjects = os.listdir(data_path)
        filter_and_ica(data_path, subjects)
        print("Filtered and ICA files saved.")

    # Plot example process of filtering
    plot_filter_process_example = False

    if plot_filter_process_example:
        # -------------------------------> I HAVE TO CHANGE THIS SO I TAKE COMPLETELY ORIGINAL DATA
        example_path = root + "Data/tDCS_EEG_data/Data_cleaned/S11/S11_post_EO.txt"
        plot_example_process("S12_post_EC.txt", "S12", 0.5, 40)
        plt.show()

    plot_artifact_removal_example = False

    if plot_artifact_removal_example:
        test_file = root + "Data/tDCS_EEG_data/Data_cleaned/S11/S11_post_EO_filt.txt"
        test_file_comps = (
            root + "Data/tDCS_EEG_data/Data_cleaned/S11/S11_post_EO_ICA_comps.txt"
        )
        test_file_mix = (
            root + "Data/tDCS_EEG_data/Data_cleaned/S11/S11_post_EO_ICA_mix.txt"
        )
        test_rec = pd.read_csv(test_file, sep="\t", index_col=False)
        test_rec_comps = pd.read_csv(test_file_comps, sep="\t", index_col=False)
        test_rec_mix = pd.read_csv(test_file_mix, sep="\t", index_col=False)
        comp_ids = [0, 14, 25]
        updated_comps, updated_rec = remove_artifacts(
            test_rec, test_rec_comps, test_rec_mix, comp_ids
        )

        plot_record(test_rec, "S11_post_EO_filt.txt")
        plot_record(test_rec_comps, "ICA - S11_post_EO_ICA_comps.txt")
        plot_record(updated_comps, "Updated ICA - S11_post_EO_ICA_comps.txt")
        plot_record(updated_rec, "Reconstructed - S11_post_EO_filt.txt")

        plt.show()

    # Loop through all filtered records and their ICA components
    # Plot to view and find which components contain artifacts to remove
    view_ICAs = False

    if view_ICAs:
        data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
        subjects = os.listdir(data_path)
        figs = 0

        for sub in subjects:
            files = os.listdir(data_path + sub + "/")
            files = sorted(files)
            for file in files:
                if "filt" in file:
                    data = pd.read_csv(
                        data_path + sub + "/" + file, sep="\t", index_col=False
                    )
                    plot_record(data, "Filtered - " + file)
                    figs += 1
                if "comps.txt" in file:
                    data = pd.read_csv(
                        data_path + sub + "/" + file, sep="\t", index_col=False
                    )
                    plot_record(data, "ICA - " + file)
                    figs += 1
                if figs == 2:
                    plt.show()
                    figs = 0

    # Fetch and save data files from second round of data obtained
    check_file = root + "Data/tDCS_EEG_data/Data_cleaned/S51/S51_post_EC.txt"
    if not os.path.exists(check_file):
        pre_d2_path = root + "Data/tDCS_EEG_data/Pre_dataset_2/"
        pre_d2 = os.listdir(pre_d2_path)
        pre_d2_files = []
        for file in pre_d2:
            if ".set" in file:
                pre_d2_files.append(pre_d2_path + file)
        clean_data(pre_d2_files)

        post_d2_path = root + "Data/tDCS_EEG_data/Post_dataset_2/"
        post_d2 = os.listdir(post_d2_path)
        post_d2_files = []
        for file in post_d2:
            if ".set" in file:
                post_d2_files.append(post_d2_path + file)
        clean_data(post_d2_files)

    # Then filter and run ICA on the new data
    # Again, we only want to run this ONCE so we are always using the same ICA
    check_file = root + "Data/tDCS_EEG_data/Data_cleaned/S54/S54_pre_EO_ICA_mix.txt"
    if not os.path.exists(check_file):
        data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
        subjects = new_data
        filter_and_ica(data_path, subjects)
        print("Filtered and ICA files saved.")

    # Get together all "filt" files of all subjects
    all_filt_files = []
    data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
    subjects = os.listdir(data_path)
    for sub in subjects:
        if sub == ".DS_Store":  # This file keeps creeping in, skip it
            continue
        files = os.listdir(data_path + sub + "/")
        for file in files:
            if "filt" in file:
                all_filt_files.append(file)
    all_filt_files = sorted(all_filt_files)

    # Now, we use the ICA and remove all the unwanted artifacts we found with manual viewing from the records
    check_file = root + "Data/tDCS_EEG_data/Data_ready/S1_post_EC_ready.txt"
    if not os.path.exists(check_file):
        comps_to_remove = pd.read_csv("Data/Components_to_remove.csv", index_col=False)
        print("Starting artifact removal.")
        i = 0

        for file in all_filt_files:
            # Get the indices of the components to remove
            tab_vals = comps_to_remove.loc[
                comps_to_remove["Filt_filename"] == file
            ].iloc[:, 1:5]
            comp_list = tab_vals.values.tolist()[0]
            comp_ids = [int(val) for val in comp_list if not m.isnan(val)]

            # Get the dataframes of the filtered record, ICA component and mixture matrices for calculations
            subject = re.split("_", file)[0]
            data_path = root + "Data/tDCS_EEG_data/Data_cleaned/" + subject + "/"
            record = pd.read_csv(data_path + file, sep="\t", index_col=False)
            ica_comps = pd.read_csv(
                data_path + file[:-8] + "ICA_comps.txt", sep="\t", index_col=False
            )
            ica_mixing = pd.read_csv(
                data_path + file[:-8] + "ICA_mix.txt", sep="\t", index_col=False
            )

            # Remove artifacts and save updated records
            updated_comps, updated_rec = remove_artifacts(
                record, ica_comps, ica_mixing, comp_ids
            )
            updated_comps.to_csv(
                data_path + file[:-8] + "ICA_comps_updated.txt", sep="\t", index=False
            )
            updated_rec.to_csv(
                "Data/tDCS_EEG_data/Data_ready/" + file[:-8] + "ready.txt",
                sep="\t",
                index=False,
            )

            i += 1
            print(i, "/", len(all_filt_files))

        print("All artifact removals finished and saved.")

    # Cut the beginning and end of the recordings
    # We cut off the first 4 seconds and then after 120 seconds
    if not os.listdir(root + "Data/tDCS_EEG_data/Epochs/116_seconds/"):
        data_path = root + "Data/tDCS_EEG_data/Data_ready/"
        files = os.listdir(data_path)
        print("Performing time-cutoffs")
        i = 0
        for file in files:
            record = pd.read_csv(data_path + file, sep="\t", index_col=False)
            rec_cut_116s = cut_record_116s(record)

            # Form filename
            name_split = file[:-4].split("_")
            sub_idx = name_split[0].find("S")
            subject = name_split[0][sub_idx:]
            h_or_d = ""
            if subject in healthy:
                h_or_d = "Healthy_"
            elif subject in depressed_active:
                h_or_d = "DepActive_"
            else:
                h_or_d = "DepSham_"

            # rec_cut_116s = pd.to_numeric(rec_cut_116s)
            rec_cut_116s.to_csv(
                "Data/tDCS_EEG_data/Epochs/116_seconds/"
                + file[:-9]
                + h_or_d
                + "116s.txt",
                sep="\t",
                index=False,
            )
            i += 1
            print(i, "/", len(files))

    # Save all "ready" plots to review them
    save_plots = False

    if save_plots:
        data_path = root + "Data/tDCS_EEG_data/Epochs/116_seconds/"
        files = os.listdir(data_path)
        i = 0
        print("Saving figures")

        for filename in files:
            ready_data = pd.read_csv(data_path + filename, sep="\t", index_col=False)
            plot_record(ready_data, filename)
            # plt.savefig("Images/" + filename[:-4])
            plt.savefig(
                "C:/Users/erlahrafnkels/Pictures/DepEEG_data_ready/" + filename[:-4]
            )
            i += 1
            print(i)
