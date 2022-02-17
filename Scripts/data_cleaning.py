import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


# Global variables
# OBS: The T3 electrode is sometimes referred to A2 instead of A1
# We don't let it bother us for now as it likely doesn't have much effect on calculations
# fmt: off
electrodes_arranged = [
    'Fp1-A1', 'Fpz-A1', 'Fp2-A2', 'F7-A1', 'F3-A1', 'Fz-A1', 'F4-A2', 'F8-A2', 'FT7-A1', 'FC3-A1', 'FCz-A1', 'FC4-A2',
    'FT8-A2', 'C3-A1', 'Cz-A2', 'C4-A2', 'T4-A2', 'TP7-A1', 'CP3-A1', 'CPz-A2', 'CP4-A2', 'TP8-A2','T5-A1', 'P3-A1',
    'Pz-A2', 'P4-A2', 'T6-A2', 'O1-A1', 'Oz-A2', 'O2-A2',
]
els_with_t3a1 = electrodes_arranged.copy()
els_with_t3a1.insert(13, 'T3-A1')
els_with_t3a2 = electrodes_arranged.copy()
els_with_t3a2.insert(13, 'T3-A2')
A1ref = ["F7", "FT7", "T3", "TP7", "T5", "Fp1", "F3", "FC3", "C3", "CP3", "P3", "O1", "Fpz", "Fz", "FCz"]
A2ref = ["Cz", "CPz", "Pz", "Oz", "Fp2", "F4", "FC4", "C4", "CP4", "P4", "O2", "F8", "FT8", "T4", "TP8", "T6"]
# fmt: on


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
    df.update(df[A1ref].sub(df["A1"], axis=0))
    df.columns = [x + "-A1" if x in A1ref else x for x in df]
    df.update(df[A2ref].sub(df["A2"], axis=0))
    df.columns = [x + "-A2" if x in A2ref else x for x in df]
    return df


def clean_data():
    files = get_files()
    for file in files:
        df = pd.read_csv(file, sep="\t", index_col=False)

        # First, subtract A1 and A2 references from corresponding columns if it hasn't been done
        if not ("-A1" or "-A2") in df.columns[0]:
            df = correct_refs(df)

        # Then, remove all unnecessary columns and arrange correctly
        if 'T3-A1' in df.columns:
            df = df[df.columns.intersection(els_with_t3a1)]
            df = df[els_with_t3a1]
        else:
            df = df[df.columns.intersection(els_with_t3a2)]
            df = df[els_with_t3a2]

        # Then, save in new file with _clean in "Depressed_cleaned" or "Healthy_cleaned" folders

    return


if __name__ == "__main__":
    clean_data()
