import os

import pandas as pd


# Global variables
# fmt: off
healthy = [
    "S2", "S3", "S9", "S10", "S12", "S13", "S14", "S15", "S19", "S20", "S24", "S25", "S30", "S32", "S38", "S39", "S42",
    "S46", "S29", "S6", "S23", "S47", "S49"
]
# fmt: on
depressed_active = ["S1", "S4", "S5", "S7", "S8", "S11", "S16", "S17", "S18", "S21", "S22", "S26", "S27"]
depressed_sham = ["S31", "S33", "S35", "S36", "S37", "S40", "S41", "S43", "S44", "S45"]
depressed = depressed_active + depressed_sham  # Delete if unused
A1ref = ["F7", "FT7", "T3", "TP7", "T5", "Fp1", "F3", "FC3", "C3", "CP3", "P3", "O1", "Fpz", "Fz", "FCz"]
A2ref = ["Cz", "CPz", "Pz", "Oz", "Fp2", "F4", "FC4", "C4", "CP4", "P4", "O2", "F8", "FT8", "T4", "TP8", "T6"]


def get_files():
    data_path = "/Users/erlahrafnkelsdottir/Documents/DepEEG/Data/tDCS_EEG_data/"
    # data_path = "Data/tDCS_EEG_data/"
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
    df.update(df[A1ref].sub(df['A1'], axis=0))
    df.columns = [x + '-A1' if x in A1ref else x for x in df]
    df.update(df[A2ref].sub(df["A2"], axis=0))
    df.columns = [x + '-A2' if x in A2ref else x for x in df]
    return df


def clean_data():
    files = get_files()
    for file in files:
        df = pd.read_csv(file, sep='\t', index_col=False)

        # First, subtract A1 and A2 from corresponding columns if it hasn't been done already
        if not ("-A1" or "-A2") in df.columns[0]:
            print(file)
            df = correct_refs(df)

        # Then, delete all unnecessary columns
        A1cols = [s + "-A1" for s in A1ref]
        A2cols = [s + "-A2" for s in A2ref]
        all_cols = A1cols + A2cols
        df_final = df.drop(columns=df.columns.difference(all_cols))
        # df = df[df.columns.intersection(all_cols)]
        df_final.sort_index(axis=1)
        # if df.shape[1] >= 31:
        #     print(df.columns)
        print(df_final.shape)

        # Then, order all columns identically

        # Then, save in new file with _clean in "Depressed_cleaned" or "Healthy_cleaned" folders
    return


A1cols = [s + "-A1" for s in A1ref]
A2cols = [s + "-A2" for s in A2ref]
all_cols = A1cols + A2cols
print(len(all_cols))
print(all_cols)


"""
def correct_refs():
    files = get_files()
    print("------------------------- START -------------------------")
    for file in files:
        df = pd.read_csv(file, sep='\t', nrows=0)
        headers = list(df)
        if headers[0] != 'Fp1-A1':
            #if 'A1' not in headers:
            df.sort_index(axis=1)
            print()
            print(os.path.basename(file))
            print("---")
            print(headers)
            print()
    print("-------------------------- END --------------------------")


def subtract_ref(df):
    if ("A1" or "A2") not in df.columns[0]:
        return


def filter_columns(df):
    # Replace all zeros with NaNs
    data_s1_eo1 = data_s1_eo1.replace(0,np.nan)
    data_s1_eo1 = data_s1_eo1.dropna(axis=1, thresh=2)
    data_s1_eo1 = data_s1_eo1.drop(["AUX1","AUX2","AUX3","AUX4"], axis=1)
    print(data_s1_eo1.shape)
    return

# Remove function if I don't need it
def get_channels():
    file = "Data/Standard-10-10-Cap31-eeg.txt"
    cap_sys = pd.read_csv(file, sep="\t")
    channels = np.array(cap_sys["labels"].values.tolist())
    return channels

# Cleaning all data
files = get_files()

print(len(files))

file1 = files[0]
print(file1)
data_s1_ec1 = pd.read_csv(file1, sep='\t', index_col=False)
data_s1_ec1.head()
print(data_s1_ec1.iloc[0,-1])

#if ("A1" or "A2") not in df.columns[0]:
"""
