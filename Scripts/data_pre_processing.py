import pandas as pd
import numpy as np

# Global variables
base_path = "G:/My Drive/Skoli/DTU/Thesis/DepEEG/"
healthy = ["S2","S3","S9","S10","S12","S13","S14","S15","S19","S20","S24","S25","S30","S32","S38","S39","S42","S46","S29","S6","S23","S47","S49"]
depressed_active = ["S1","S4","S5","S7","S8","S11","S16","S17","S18","S21","S22","S26","S27"]
depressed_sham = ["S31","S33","S35","S36","S37","S40","S41","S43","S44","S45"]
depressed = depressed_active + depressed_sham

def get_channels():
    file = base_path + "Data/Standard-10-10-Cap31-eeg.txt"
    cap_sys = pd.read_csv(file, sep="\t")
    channels = np.array(cap_sys["labels"].values.tolist())
    return channels
