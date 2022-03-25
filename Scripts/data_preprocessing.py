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
healthy = [
    "S2", "S3", "S9", "S10", "S12", "S13", "S14", "S15", "S19", "S20", "S24", "S25", "S30", "S32", "S38", "S39", "S42",
    "S46", "S29", "S6", "S23", "S47", "S49",
]
depressed_active = ["S1", "S4", "S5", "S7", "S8", "S11", "S16", "S17", "S18", "S21", "S22", "S26", "S27"]
depressed_sham = ["S31", "S33", "S35", "S36", "S37", "S40", "S41", "S43", "S44", "S45"]
depressed = depressed_active + depressed_sham
# fmt: on
samp_freq = 500  # Sample frequency, given in data description (Hz)
config = OmegaConf.load("config.yaml")
color_codes = [c[1] for c in config.colors.items()]

# Get root folder based on which operating system I'm working on (needed for mac)
root = ""
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/"


def get_files() -> list:
    data_path = root + "Data/tDCS_EEG_data/"
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

    return new_filename


def correct_refs(df: pd.DataFrame) -> pd.DataFrame:
    df.update(df[A1ref].sub(df["A1"], axis=0))
    df.columns = [x + "-A1" if x in A1ref else x for x in df]
    df.update(df[A2ref].sub(df["A2"], axis=0))
    df.columns = [x + "-A2" if x in A2ref else x for x in df]
    return df


def clean_data():
    files = get_files()
    print("Cleaning data files.")
    for file in files:
        df = pd.read_csv(file, sep="\t", index_col=False)

        # First, subtract A1 and A2 references from corresponding columns if it hasn't been done
        if not ("-A1" or "-A2") in df.columns[0]:
            df = correct_refs(df)

        # Then, remove all unnecessary columns and arrange correctly
        if "T3-A1" in df.columns:
            df = df[df.columns.intersection(els_with_t3a1)]
            df = df[els_with_t3a1]
        else:
            df = df[df.columns.intersection(els_with_t3a2)]
            df = df[els_with_t3a2]

        # Update filename and save
        filename = update_filename(file)
        df.to_csv(
            root + "Data/tDCS_EEG_data/Data_cleaned/" + filename, sep="\t", index=False
        )

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


def remove_eye_blinks(record: pd.DataFrame, filename: str):  # -> pd.DataFrame:
    # np.polyfit(x, y, deg)

    # Set up x-axis in time domain
    points = record.shape[0]
    x = np.linspace(0, points / samp_freq, points)

    chanFp1 = record["Fp1-A1"]

    max_peak = chanFp1.max()
    max_pos = chanFp1.idxmax()
    det_window = [0.25 * max_peak, max_peak]

    start_points = []
    temp = 0
    for idx, p in chanFp1.iteritems():
        if p >= det_window[0]:
            if temp == 0:
                start_points.append((idx / samp_freq, p))
                temp = 1
        else:
            temp = 0

    print("Number of blinks detected: ", len(start_points))

    fig = plt.figure(figsize=(14, 6))
    plt.axhline(y=0, color="black")
    plt.plot(x, chanFp1)
    plt.plot(max_pos / samp_freq, max_peak, marker="o", color="r")
    plt.axhline(y=det_window[0], color="r")
    for p in start_points:
        plt.plot(p[0], p[1], marker="o", color="g")
    # plt.axhline(y=det_window[1], color="r")
    plt.title(make_plot_title(filename), fontsize="x-large")

    return fig


def filter_record(filename: str, low_freq: float, high_freq: float) -> pd.DataFrame:
    # Read in raw, cleaned data
    data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
    data = pd.read_csv(data_path + filename, sep="\t", index_col=False)

    # For each channel apply filters, normalize and replace data column values with new
    for col in data.columns:
        channel = data[col]
        _, _, signal_notched = notch_filter(channel, low_freq, high_freq)
        signal_filtered = butter_bandpass_filter(signal_notched, low_freq, high_freq)
        sf_norm = zscore(signal_filtered)
        data[col] = sf_norm

    # if "EO" in filename:
    #    remove_eye_blinks(data, filename)  # Produces a figure
    #    plt.show()

    return data


def perform_ICA(record: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame]:
    ica = FastICA(random_state=42)
    comps_ = ica.fit_transform(record)  # Reconstruct signals (type: np.ndarray)
    mixing_ = ica.mixing_  # Get estimated mixing matrix (type: np.ndarray)

    # Convert from numpy array to pandas dataframe
    comps = pd.DataFrame(comps_)
    mixing = pd.DataFrame(mixing_)

    # plt.figure("rec")
    # plt.plot(record)
    # for i in range(S.shape[1]):
    #     plt.figure(i)
    #     plt.plot(S.iloc[i,:])
    # plt.show()

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
    subject = name_split[0]
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

    plot_title = subject + ": " + h_or_d + pre_or_post + open_or_closed

    return plot_title


def plot_example_process(
    filename: str, low_freq: float, high_freq: float
) -> plt.figure:
    # Read in raw, cleaned data
    data_path = root + "Data/tDCS_EEG_data/Data_cleaned/"
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


def plot_record(record: pd.DataFrame, filename: str) -> plt.figure:
    # Set up x-axis in time domain
    channels = record.shape[1]
    datapoints = record.shape[0]
    x = np.linspace(0, datapoints / samp_freq, datapoints)

    # Compose plot
    fig = plt.figure(figsize=(12, 8))
    color = 0

    for i in range(0, channels):
        if "ICA" in filename:
            y = i / 10
        else:
            y = (
                i * 20
            )  # Placement of signal on y-axis, stack channels one after the other
            plt.text(-1, -y - 3, record.columns[i], fontsize="small", ha="right")
        if color == len(color_codes):
            color = 0
        plt.plot(x, record.iloc[:, i] - y, color=color_codes[color])
        color += 1

    plt.title(make_plot_title(filename), fontsize="x-large")
    plt.xlabel("Time [s]")
    plt.yticks([])
    plt.box(False)
    plt.grid(color="#D6D6D6")
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Run this if we have don't have the cleaned files
    if not os.listdir(root + "Data/tDCS_EEG_data/Data_cleaned/"):
        clean_data()
        print("Cleaned files saved.")

    # fig1 = plot_example_process("S32_pre_EC.txt", 0.5, 50)
    # fig1 = plot_example_process("S32_H_post_EC.txt", 0.5, 50)
    # plt.show()
    # filter_record("S1-Pre-EC1_EEG_cleaned.txt", 0.5, 50)

    # Example of bad: 36, 12
    subject_no = 12

    record_names = [
        "S" + str(subject_no) + "_pre_EO.txt",
        "S" + str(subject_no) + "_pre_EC.txt",
        "S" + str(subject_no) + "_post_EO.txt",
        "S" + str(subject_no) + "_post_EC.txt",
    ]
    figs = []
    records = []
    for name in record_names:
        record_filtered = filter_record(name, 0.5, 50)
        records.append(record_filtered)
        # figs.append(plot_record(record_filtered, name))

    rec1 = records[0]
    # channel1 = rec1.iloc[:, 0]
    # plt.figure("1channel")
    # figchan = plt.plot(channel1)
    # figs.append(figchan)

    # plt.show()

    # recs_for_ica = records[0].iloc[:, 0:3]

    # comps, mixing = perform_ICA(recs_for_ica)

    # plot_record(recs_for_ica, record_names[0])
    # plot_record(comps, "ICA - "+ record_names[0])

    # plt.show()

    mne_top(rec1)
