import math as m
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import signal

# from scipy.stats import zscore

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


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.filtfilt(b, a, data)
    return y


def filter():
    data_path = "Data/tDCS_EEG_data/Data_cleaned/"
    s1_eo1_data = pd.read_csv(
        data_path + "S1-Pre-EO1_EEG_cleaned.txt", sep="\t", index_col=False
    )
    """
    s1_ec1_data = pd.read_csv(
        data_path + "S1-Pre-EC1_EEG_cleaned.txt", sep="\t", index_col=False
    )
    s2_eo1_data = pd.read_csv(
        data_path + "S2-Pre_EO_EEG_cleaned.txt", sep="\t", index_col=False
    )
    s2_ec1_data = pd.read_csv(
        data_path + "S2-Pre-_EC1_EEG_cleaned.txt", sep="\t", index_col=False
    )
    """

    # Plot the raw signal
    plt.figure("raw")
    plt.plot(s1_eo1_data.iloc[:, 0], color=config.colors.dtu_red)
    plt.title("Example of raw data: Eyes open pre-treatment")
    plt.grid()

    # Upper and lower bounds for bandpass filter
    low_freq = 0.5
    high_freq = 50

    # Create notch filter
    samp_freq = 500  # Sample frequency, given in data description (Hz)
    notch_freq = 50  # Frequency to be removed from signal (Hz)
    quality_factor = (high_freq - low_freq) / m.sqrt(low_freq * high_freq)
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    # Outputs of signal.freqz below:
    freq, h = signal.freqz(b_notch, a_notch, fs=samp_freq)

    # Plot notch filter
    # In the plot we transform h from complex numbers to dBs
    plt.figure("notch filter")
    plt.plot(freq, 20 * np.log10(abs(h)), color=config.colors.dtu_red)
    plt.title("Notch filter")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid()

    # Filter out the 50 Hz powerline interference
    signal_notched = signal.filtfilt(b_notch, a_notch, s1_eo1_data.iloc[:, 0])

    # Apply bandpass filter
    signal_filtered = butter_bandpass_filter(
        signal_notched, low_freq, high_freq, samp_freq
    )  # , order=6)

    # Plot the power spectral density of the raw, notched and filtered signals
    plt.figure("psd")
    plt.psd(s1_eo1_data.iloc[:, 0], Fs=samp_freq, color=config.colors.dtu_red)
    plt.psd(signal_notched, Fs=samp_freq, color=config.colors.black)
    plt.psd(signal_filtered, Fs=samp_freq, color=config.colors.blue)
    plt.legend(labels=["Raw", "Notched", "Bandpass + notched"])
    plt.title("Power spectral density of example")

    # Plot the original signal with the filters
    plt.figure("bandpass")
    plt.plot(signal_filtered, color=config.colors.dtu_red)
    plt.title("Example filtered")
    plt.grid()

    plt.show()

    return


if __name__ == "__main__":
    # We don't want to run if we have already made the cleaned files
    if not os.listdir("Data/tDCS_EEG_data/Data_cleaned/"):
        clean_data()
        print("Cleaned files saved.")
    else:
        print("Cleaned files in folder.")

    filter()
