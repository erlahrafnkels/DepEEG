import pandas as pd
import numpy as np
import os

# Global variables
#base_path = "G:/My Drive/Skoli/DTU/Thesis/DepEEG/"
healthy = ["S2","S3","S9","S10","S12","S13","S14","S15","S19","S20","S24","S25","S30","S32","S38",
           "S39","S42","S46","S29","S6","S23","S47","S49"]
depressed_active = ["S1","S4","S5","S7","S8","S11","S16","S17","S18","S21","S22","S26","S27"]
depressed_sham = ["S31","S33","S35","S36","S37","S40","S41","S43","S44","S45"]
depressed = depressed_active + depressed_sham

def get_channels():
    #file = base_path + "Data/Standard-10-10-Cap31-eeg.txt"
    file = "Data/Standard-10-10-Cap31-eeg.txt"
    cap_sys = pd.read_csv(file, sep="\t")
    channels = np.array(cap_sys["labels"].values.tolist())
    return channels

def get_files():
    data_path = 'Data/tDCS_EEG_data/'#'G:/My Drive/Skoli/DTU/Thesis/DepEEG/Data/tDCS_EEG_data/'
    subject_folders = os.listdir(data_path)
    txt_file_paths = []

    for subject in subject_folders:
        path = data_path + subject
        subfolders = os.listdir(path)
        for p_folders in subfolders:
            str = path + '/' + p_folders
            pre_post = os.listdir(str)
            for file in pre_post:
                file_path = str + '/' + file
                txt_file_paths.append(file_path)

    return txt_file_paths

files = get_files()
print(files)