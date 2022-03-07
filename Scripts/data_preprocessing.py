import math as m
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import signal
from scipy.stats import zscore

warnings.filterwarnings("ignore")


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
samp_freq = 500  # Sample frequency, given in data description (Hz)
# fmt: on
config = OmegaConf.load("config.yaml")


def get_files():
    data_path = "Data/tDCS_EEG_data/"
    subject_folders = os.listdir(data_path)
    txt_file_paths = []

    for subject in subject_folders:
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


def correct_refs(df):
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

        # Then, save in new file with _cleaned in "Data_cleaned" folder
        filename = os.path.basename(file)[:-4] + "_cleaned.txt"
        df.to_csv("Data/tDCS_EEG_data/Data_cleaned/" + filename, sep="\t", index=False)

    return


def notch_filter(data, low_freq, high_freq):
    notch_freq = 50  # Frequency to be removed from signal (Hz)
    quality_factor = (high_freq - low_freq) / m.sqrt(low_freq * high_freq)
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
    signal_notched = signal.filtfilt(b_notch, a_notch, data)
    return freq, h, signal_notched


def butter_bandpass_filter(data, low_freq, high_freq, order=5):
    nyq = 0.5 * samp_freq
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.filtfilt(b, a, data)
    return y


def filter_signal(record):
    # Read in data
    data_path = "Data/tDCS_EEG_data/Data_cleaned/"
    data = pd.read_csv(data_path + record, sep="\t", index_col=False)
    datanp = data.to_numpy()

    rows, cols = data.shape
    print(rows)
    print(cols)

    datareshape = np.reshape(datanp, (1, -1))
    print(datareshape.shape)
    data2d = np.reshape(datareshape, (-1, cols))

    low_freq = 0.5
    high_freq = 50

    freq, h, signal_notched = notch_filter(datareshape, low_freq, high_freq)
    signal_filtered = butter_bandpass_filter(signal_notched, low_freq, high_freq)

    filtered2d = np.reshape(signal_filtered, (-1, cols))

    plt.figure(1)
    plt.plot(data)
    plt.title("Dataframe")

    plt.figure(2)
    plt.plot(datanp)
    plt.title("Numpy array")

    plt.figure(3)
    plt.plot(data2d)
    plt.title("Numpy array reshaped")

    plt.figure(4)
    plt.plot(filtered2d)
    plt.title("Numpy array reshaped filtered")

    plt.show()

    return


def filter_and_plot(record):
    # Read in data
    data_path = "Data/tDCS_EEG_data/Data_cleaned/"
    data = pd.read_csv(data_path + record, sep="\t", index_col=False)
    # raw = data.iloc[:, 0] # Here we're only looking at the records of Fp1

    datanp = data.to_numpy()
    rows, cols = data.shape
    datareshape = np.reshape(datanp, (1, -1))

    # Upper and lower bounds for bandpass filter
    low_freq = 0.5
    high_freq = 50

    # Filter out the 50 Hz powerline interference
    freq, h, signal_notched = notch_filter(datareshape, low_freq, high_freq)

    # Apply bandpass filter
    signal_filtered = butter_bandpass_filter(signal_notched, low_freq, high_freq)

    signal_notched = np.reshape(signal_notched, (-1, cols))
    signal_filtered = np.reshape(signal_filtered, (-1, cols))

    # Make one big plot
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Raw signal
    axs[0, 0].plot(datanp[:, 0], color=config.colors.dtu_red)
    axs[0, 0].set_title("Raw data")
    axs[0, 0].grid()

    # Notch filter
    # In the plot we transform h from complex numbers to dBs
    axs[0, 1].plot(freq, 20 * np.log10(abs(h)), color=config.colors.dtu_red)
    axs[0, 1].set_title("Notch filter")
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Amplitude [dB]")
    axs[0, 1].grid()

    # Power spectral density of the raw, notched and filtered signals
    axs[1, 0].psd(datanp[:, 0], Fs=samp_freq, color=config.colors.dtu_red)
    axs[1, 0].psd(signal_notched[:, 0], Fs=samp_freq, color=config.colors.black)
    axs[1, 0].psd(signal_filtered[:, 0], Fs=samp_freq, color=config.colors.blue)
    axs[1, 0].legend(labels=["Raw", "Notched", "Bandpass + notched"])
    axs[1, 0].set_title("Power spectral density")

    # Original signal with the filters
    axs[1, 1].plot(signal_filtered[:, 0], color=config.colors.dtu_red)
    axs[1, 1].set_title("Data filtered")
    axs[1, 1].grid()

    fig.suptitle(record[:-4])
    fig.tight_layout()

    # Now, normalize data with mean = 0 and std = 1
    # Find one mean and std for all channels and the whole 2 minute recording
    print("SIGNAL RAW SHAPE:", datanp.shape)
    print("SIGNAL RAW MEAN:", datanp.mean())
    print("SIGNAL RAW STD:", datanp.std())

    print("SIGNAL FILTERED SHAPE:", signal_filtered.shape)
    print("SIGNAL FILTERED MEAN:", signal_filtered.mean())
    print("SIGNAL FILTERED STD:", signal_filtered.std())

    data_norm = zscore(signal_filtered, axis=None)
    print("DATA NORM SHAPE:", data_norm.shape)
    print("DATA NORM MEAN:", data_norm.mean())
    print("DATA NORM STD:", data_norm.std())
    print()
    print("Normalized, mean of first row:", data_norm[0, :].mean())
    print()
    print()

    mean = data.stack().mean()
    std = data.stack().std()
    data_norm2 = (data - mean) / std

    print("DATA NORM 2 MEAN:", data_norm2.mean())
    print("DATA NORM 2 STD:", data_norm2.std())
    print()
    print("Normalized 2, mean of first row:", data_norm2[0, :].mean())

    # Convert recordings to time in seconds
    # time_axis = range(0, m.floor(data.shape[0] / samp_freq))

    # Plot all channels
    fig2, axs2 = plt.subplots(nrows=31, ncols=1, sharex=True, figsize=(12, 8))
    for i in range(0, 31):
        axs2[i].plot(signal_filtered[:, i])
        # axs2[i].set_xticks(time_axis)
        axs2[i].set_frame_on(False)

    return fig, fig2


if __name__ == "__main__":
    # We don't want to run if we have already made the cleaned files
    if not os.listdir("Data/tDCS_EEG_data/Data_cleaned/"):
        clean_data()
        print("Cleaned files saved.")
    else:
        print("Cleaned files in folder.")

    fig1, fig2 = filter_and_plot("S1-Pre-EC1_EEG_cleaned.txt")
    # fig2 = filter_and_plot("S1-post_EC1_EEG_cleaned.txt")
    # fig3 = filter_and_plot("S42_pre_EC1_cleaned.txt")
    # fig4 = filter_and_plot("S42_EC1Post_EEG_cleaned.txt")
    plt.show()

    # filter_signal("S1-Pre-EC1_EEG_cleaned.txt")
