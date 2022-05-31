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
from scipy import signal
from scipy.stats import zscore
from sklearn.decomposition import FastICA

warnings.filterwarnings("ignore")


"""
ABOUT THIS SCRIPT ....
....
....

Obs: I added input and output types to all of the functions, even though it doesn't change anything in how the
function is read and executed. It was merely a way for me to remember and easily see the types; i.e. whether
I had a pandas dataframe or a numpy array, for example.

"""


# Global variables
# OBS: The T3 electrode is sometimes referred to A2 instead of A1
# We don't let it bother us for now as it likely doesn't have much effect on calculations
# fmt: off
electrodes_arranged = [
    'Fp1-A1', 'Fpz-A1', 'Fp2-A2', 'F7-A1', 'F3-A1', 'Fz-A1', 'F4-A2', 'F8-A2', 'FT7-A1', 'FC3-A1', 'FCz-A1', 'FC4-A2',
    'FT8-A2', 'C3-A1', 'Cz-A2', 'C4-A2', 'T4-A2', 'TP7-A1', 'CP3-A1', 'CPz-A2', 'CP4-A2', 'TP8-A2', 'T5-A1', 'P3-A1',
    'Pz-A2', 'P4-A2', 'T6-A2', 'O1-A1', 'Oz-A2', 'O2-A2',
]
els_with_t3a1 = electrodes_arranged.copy()
els_with_t3a1.insert(13, 'T3-A1')
els_with_t3a2 = electrodes_arranged.copy()
els_with_t3a2.insert(13, 'T3-A2')
A1ref = ["F7", "FT7", "T3", "TP7", "T5", "Fp1", "F3", "FC3", "C3", "CP3", "P3", "O1", "Fpz", "Fz", "FCz"]
A2ref = ["Cz", "CPz", "Pz", "Oz", "Fp2", "F4", "FC4", "C4", "CP4", "P4", "O2", "F8", "FT8", "T4", "TP8", "T6"]
new_data = ["S51", "S52", "S53", "S54", "S55", "S56", "S57", "S58", "S59", "S60", "S61"]
healthy = [
    "S2", "S3", "S9", "S10", "S12", "S13", "S14", "S15", "S19", "S20", "S24", "S25", "S30", "S32", "S38", "S39", "S42",
    "S46", "S29", "S6", "S23", "S47", "S49", "S53", "S55", "S56", "S57", "S60", "S61"
]
depressed_active = [
    "S1", "S4", "S5", "S7", "S8", "S11", "S16", "S17", "S18", "S21", "S22", "S26", "S27", "S51", "S52", "S54", "S58",
    "S59"
]
depressed_sham = ["S31", "S33", "S35", "S36", "S37", "S40", "S41", "S43", "S44", "S45"]
# fmt: on
samp_freq = 500  # Sample frequency, given in data description (Hz)
config = OmegaConf.load("config.yaml")
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


def mne_top(record: pd.DataFrame):
    channels = record.columns.to_list()
    info = mne.create_info(ch_names=channels, sfreq=samp_freq, ch_types="eeg")
    data = mne.io.RawArray(record.T, info)
    # data.plot(duration=5, n_channels=len(channels))

    print(channels[0:5])
    print(data.info)
    print(data.pick_channels())

    ev_data = data[0, :]
    print(len(ev_data))
    ev_info = mne.create_info(ch_names=channels[0:5], sfreq=samp_freq, ch_types="eeg")
    evoked = mne.EvokedArray(ev_data, ev_info)
    mne.viz.plot_topomap(evoked.data, evoked.info)
    plt.show()

    return


def make_plot_title(filename: str) -> str:
    name_split = filename[:-4].split("_")
    beginning = name_split[0]
    sub_idx = name_split[0].find("S")
    subject = name_split[0][sub_idx:]
    pre_or_post = name_split[1]
    open_or_closed = name_split[2]
    h_or_d = ""

    if subject in healthy:
        h_or_d = "Healthy"
    else:
        h_or_d = "Depressed"
    if pre_or_post == "pre":
        pre_or_post = ", pretreatment"
    else:
        pre_or_post = ", posttreatment"
    if open_or_closed == "EO":
        open_or_closed = ", eyes open"
    else:
        open_or_closed = ", eyes closed"

    plot_title = beginning + ": " + h_or_d + pre_or_post + open_or_closed

    return plot_title


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


def remove_artefacts(
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

    # Update record so unwanted artefacts are removed
    updated_rec = np.dot(comps_np, mix_np.T)
    updated_rec = pd.DataFrame(updated_rec, columns=columns)
    updated_comps = pd.DataFrame(comps_np)

    return updated_comps, updated_rec


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

    plot_artefact_removal_example = False

    if plot_artefact_removal_example:
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
        updated_comps, updated_rec = remove_artefacts(
            test_rec, test_rec_comps, test_rec_mix, comp_ids
        )

        plot_record(test_rec, "S11_post_EO_filt.txt")
        plot_record(test_rec_comps, "ICA - S11_post_EO_ICA_comps.txt")
        plot_record(updated_comps, "Updated ICA - S11_post_EO_ICA_comps.txt")
        plot_record(updated_rec, "Reconstructed - S11_post_EO_filt.txt")

        plt.show()

    # Loop through all filtered records and their ICA components
    # Plot to view and find which components contain artefacts to remove
    view_ICAs = True

    # S3 PRE IS MISSING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if view_ICAs:
        data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
        subjects = os.listdir(data_path)
        figs = 0
        i = 1

        for sub in subjects:
            files = os.listdir(data_path + sub + "/")
            files = sorted(files)
            for file in files:
                i += 1
                if i < 451:
                    continue
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

    # Now, we use the ICA and remove all the unwanted artefacts we found with manual viewing from the records
    comps_to_remove = pd.read_csv("Data/Components_to_remove.csv", index_col=False)
    check_file = root + "Data/tDCS_EEG_data/Data_cleaned/S1/S1_post_EC_ready.txt"
    if not os.path.exists(check_file):
        for file in all_filt_files:
            # Get the indices of the components to remove
            tab_vals = comps_to_remove.loc[
                comps_to_remove["Filt_filename"] == file
            ].iloc[:, 1:6]
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

            # Remove artefacts and save updated records
            updated_comps, updated_rec = remove_artefacts(
                record, ica_comps, ica_mixing, comp_ids
            )
            updated_comps.to_csv(
                data_path + file[:-8] + "ICA_comps_updated.txt", sep="\t", index=False
            )
            updated_rec.to_csv(
                data_path + file[:-8] + "ready.txt", sep="\t", index=False
            )

            print(file, " done")

        print("All artefact removals finished and saved.")

    # Save all "ready" plot
    save_plots = False

    if save_plots:
        data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
        subjects = os.listdir(data_path)
        i = 1
        print("Saving figures")

        for sub in subjects:
            files = os.listdir(data_path + sub + "/")
            ready_files = [f for f in files if f.__contains__("ready")]
            for filename in ready_files:
                ready_data = pd.read_csv(
                    data_path + sub + "/" + filename, sep="\t", index_col=False
                )
                plot_record(ready_data, filename)
                plt.savefig("Images/" + filename[:-4])
                print(i)
                i += 1

    """
    if save_plots:
        data_path = root + "Data/tDCS_EEG_data/Data_ready/"
        datafiles = os.listdir(data_path)
        i = 1

        for filename in datafiles:
            ready_data = pd.read_csv(data_path + filename, sep="\t", index_col=False)
            plot_record(ready_data, filename)
            plt.show()
            #plt.savefig("Images/" + filename[:-4])
            #print(i)
            #i += 1

    data3 = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S10/S10_post_EC_filt.txt", sep="\t", index_col=False)
    data4 = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S10/S10_post_EC_ready.txt", sep="\t", index_col=False)
    print("Filtered mean:", data3.stack().mean())
    print("Ready mean:", data4.stack().mean())

    data1 = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S22/S22_post_EO_ICA_comps.txt", sep="\t", index_col=False)
    data2 = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S22/S22_post_EO_ICA_comps_updated.txt", sep="\t",
        index_col=False)
    data3 = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S22/S22_post_EO_filt.txt", sep="\t", index_col=False)
    data4 = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S22/S22_post_EO_ready.txt", sep="\t", index_col=False)
    plot_record(data1, "ICA - S22_post_EO_ICA_comps.txt")
    plot_record(data2, "ICA updated - S22_post_EO_ICA_comps_updated.txt")
    plot_record(data3, "Filtered - S22_post_EO_filt.txt")
    plot_record(data4, "Ready - S22_post_EO_ready.txt")
    plt.show()

    s30_pre_ec_raw = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S30/S30_pre_EC_filt.txt", sep="\t", index_col=False)
    s30_pre_eo_raw = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S30/S30_pre_EO_filt.txt", sep="\t", index_col=False)
    s30_post_ec_raw = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S30/S30_post_EC_filt.txt", sep="\t", index_col=False)
    s30_post_eo_raw = pd.read_csv("Data/tDCS_EEG_data/Data_cleaned/S30/S30_post_EO_filt.txt", sep="\t", index_col=False)

    print("S30 pre EC mean: \t", s30_pre_ec_raw.stack().mean())
    print("S30 post EC mean: \t", s30_post_ec_raw.stack().mean())
    print("S30 pre EO mean: \t", s30_pre_eo_raw.stack().mean())
    print("S30 post EO mean: \t", s30_post_eo_raw.stack().mean())
    """
