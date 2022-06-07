import os
from sys import platform

# import numpy as np
# import pandas as pd
# import sklearn
from omegaconf import OmegaConf

# Get global variables and lists from the configuration file
config = OmegaConf.load("config.yaml")
samp_freq = config.sample_frequency
noisy_recs = config.noisy_recs

# Get root folder based on which operating system I'm working on
root = ""
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/"

# List of ready files
recs116s = os.listdir(root + "Data/tDCS_EEG_data/Epochs/116_seconds")
recs116s = sorted(list(set(recs116s) - set(noisy_recs)))


# def make_epochs(record: pd.DataFrame, seconds: int):
#    return record


if __name__ == "__main__":
    # Just dummy code
    print(recs116s)
