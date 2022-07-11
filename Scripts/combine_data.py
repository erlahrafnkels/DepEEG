import os
import pickle
from sys import platform

import pandas as pd
from omegaconf import OmegaConf

# Get stuff from configuration file
config = OmegaConf.load("config.yaml")
remove_recs = config.noisy_recs
all_electrodes = config.electrodes.all_electrodes_arranged
channel_names = [c[:-3] for c in all_electrodes]
healthy_ids = config.subject_classes.healthy_num
depressed_ids = config.subject_classes.depressed_num

# Get root folder based on which operating system I'm working on
root = "Data/tDCS_EEG_data/Epochs/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root

# Lists of ready files
recswhole = os.listdir(root + "Whole_rec/")
recswhole = [r for r in recswhole if "S" in r]
recswhole = sorted(list(set(recswhole) - set(remove_recs)))
recswhole_EC_pre = [r for r in recswhole if ("EC" in r) and ("pre" in r)]
recswhole_EO_pre = [r for r in recswhole if ("EO" in r) and ("pre" in r)]
recswhole_EC_post = [r for r in recswhole if ("EC" in r) and ("post" in r)]
recswhole_EO_post = [r for r in recswhole if ("EO" in r) and ("post" in r)]


def get_subject_id(filename):
    name_split = filename.split("_")
    subject = name_split[0][1:]
    id = int(subject)
    return id


def create_combined_dfs():
    # Let's create some combined dataframes
    # -------------- EC PRE combined dataframe --------------

    # Initialize
    first_EC_pre = recswhole_EC_pre[0]
    first_EC_pre_id = get_subject_id(first_EC_pre)
    first_EC_pre_dep = 1 if "D" in first_EC_pre else 0
    all_pre_EC_2min = pd.read_csv(
        root + "Whole_rec/" + first_EC_pre, sep="\t", index_col=False
    )
    all_pre_EC_2min["Subject_ID"] = first_EC_pre_id
    all_pre_EC_2min["Depressed"] = first_EC_pre_dep

    # Then append all other records with the same categorization
    for file in recswhole_EC_pre:
        if file == recswhole_EC_pre[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "Whole_rec/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_pre_EC_2min = all_pre_EC_2min.append(record)

    # Save as .pickle
    with open(root + "Whole_rec" + "/all_pre_EC_2min.pickle", "wb") as f:
        pickle.dump(all_pre_EC_2min, f)

    print("EC PRE combined dataframe SAVED.")

    # -------------- EO PRE combined dataframe --------------

    # Initialize
    first_EO_pre = recswhole_EO_pre[0]
    first_EO_pre_id = get_subject_id(first_EO_pre)
    first_EO_pre_dep = 1 if "D" in first_EO_pre else 0
    all_pre_EO_2min = pd.read_csv(
        root + "Whole_rec/" + first_EO_pre, sep="\t", index_col=False
    )
    all_pre_EO_2min["Subject_ID"] = first_EO_pre_id
    all_pre_EO_2min["Depressed"] = first_EO_pre_dep

    # Then append all other records with the same categorization
    for file in recswhole_EO_pre:
        if file == recswhole_EO_pre[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "Whole_rec/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_pre_EO_2min = all_pre_EO_2min.append(record)

    # Save as .pickle
    with open(root + "Whole_rec" + "/all_pre_EO_2min.pickle", "wb") as f:
        pickle.dump(all_pre_EO_2min, f)

    print("EO PRE combined dataframe SAVED.")

    # -------------- EC POST combined dataframe --------------

    # Initialize
    first_EC_post = recswhole_EC_post[0]
    first_EC_post_id = get_subject_id(first_EC_post)
    first_EC_post_dep = 1 if "D" in first_EC_post else 0
    all_post_EC_2min = pd.read_csv(
        root + "Whole_rec/" + first_EC_post, sep="\t", index_col=False
    )
    all_post_EC_2min["Subject_ID"] = first_EC_post_id
    all_post_EC_2min["Depressed"] = first_EC_post_dep

    # Then append all other records with the same categorization
    for file in recswhole_EC_post:
        if file == recswhole_EC_post[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "Whole_rec/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_post_EC_2min = all_post_EC_2min.append(record)

    # Save as .pickle
    with open(root + "Whole_rec" + "/all_post_EC_2min.pickle", "wb") as f:
        pickle.dump(all_post_EC_2min, f)

    print("EC POST combined dataframe SAVED.")

    # -------------- EO POST combined dataframe --------------

    # Initialize
    first_EO_post = recswhole_EO_post[0]
    first_EO_post_id = get_subject_id(first_EO_post)
    first_EO_post_dep = 1 if "D" in first_EO_post else 0
    all_post_EO_2min = pd.read_csv(
        root + "Whole_rec/" + first_EO_post, sep="\t", index_col=False
    )
    all_post_EO_2min["Subject_ID"] = first_EO_post_id
    all_post_EO_2min["Depressed"] = first_EO_post_dep

    # Then append all other records with the same categorization
    for file in recswhole_EO_post:
        if file == recswhole_EO_post[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "Whole_rec/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_post_EO_2min = all_post_EO_2min.append(record)

    # Save as .pickle
    with open(root + "Whole_rec" + "/all_post_EO_2min.pickle", "wb") as f:
        pickle.dump(all_post_EO_2min, f)

    print("EO POST combined dataframe SAVED.")

    # -------------- ALL PRE combined dataframe --------------

    with open(root + "Whole_rec/all_pre_EC_2min.pickle", "rb") as f:
        pre_ec = pickle.load(f)

    with open(root + "Whole_rec/all_pre_EO_2min.pickle", "rb") as f:
        pre_eo = pickle.load(f)

    pre_ec["EO"] = 0
    pre_eo["EO"] = 1

    all_pre = pd.concat([pre_ec, pre_eo], axis=0, ignore_index=True)

    # Save as .pickle
    with open(root + "Whole_rec/all_pre_2min.pickle", "wb") as f:
        pickle.dump(all_pre, f)

    # -------------- ALL POST combined dataframe --------------

    with open(root + "Whole_rec/all_post_EC_2min.pickle", "rb") as f:
        post_ec = pickle.load(f)

    with open(root + "Whole_rec/all_post_EO_2min.pickle", "rb") as f:
        post_eo = pickle.load(f)

    post_ec["EO"] = 0
    post_eo["EO"] = 1

    all_post = pd.concat([post_ec, post_eo], axis=0, ignore_index=True)

    # Save as .pickle
    with open(root + "Whole_rec/all_post_2min.pickle", "wb") as f:
        pickle.dump(all_post, f)

    # -------------- EVERYTHING IN ONE DATAFRAME --------------

    # Add pre/post and EO/EC columns
    all_pre_EC_2min["Post"] = 0
    all_pre_EC_2min["EO"] = 0
    all_pre_EO_2min["Post"] = 0
    all_pre_EO_2min["EO"] = 1
    all_post_EC_2min["Post"] = 1
    all_post_EC_2min["EO"] = 0
    all_post_EC_2min["Post"] = 1
    all_post_EC_2min["EO"] = 1

    # Combine
    all_2min = all_post_EC_2min
    all_2min = all_2min.append(all_post_EO_2min)
    all_2min = all_2min.append(all_pre_EC_2min)
    all_2min = all_2min.append(all_pre_EO_2min)

    # Save as .pickle
    with open(root + "Whole_rec" + "/all_2min.pickle", "wb") as f:
        pickle.dump(all_2min, f)

    return


def combine_BIMF_dfs():
    # Get file names
    path = root + "10_seconds/BIMFs/"
    bimf_files = os.listdir(path)
    bimf_files.remove("all_BIMFs_pre_EC.pickle")
    print(len(bimf_files))
    print(bimf_files)

    # Initialize dataframe
    with open(path + bimf_files[0], "rb") as f:
        first_file = pickle.load(f)
    all_BIMFs_df = first_file[first_file.columns[0:5]]
    bimf_files = bimf_files[1:]

    # Concatenate all other files
    for file in bimf_files:
        with open(path + file, "rb") as f:
            current_file = pickle.load(f)
        bimf_cols = current_file.columns[0:5]
        current_file = current_file[bimf_cols]
        all_BIMFs_df = pd.concat([all_BIMFs_df, current_file], axis=1)

    # Add ID columns
    id_cols = first_file[first_file.columns[-2:]]
    all_BIMFs_df = pd.concat([all_BIMFs_df, id_cols], axis=1)
    all_BIMFs_df["Depressed"] = all_BIMFs_df.apply(
        lambda row: 1 if row["Subject_ID"] in depressed_ids else 0, axis=1
    )

    # Save as .pickle
    with open(path + "/all_BIMFs_post_EC.pickle", "wb") as f:
        pickle.dump(all_BIMFs_df, f)

    print("Combined BIMF dataframe SAVED.")

    return


if __name__ == "__main__":
    # Check whether we have already made and saved the combined data files
    check_file = root + "Whole_rec/all_2min.pickle"
    if not os.path.exists(check_file):
        create_combined_dfs()

    # Check whether we have already made and saved the combined BIMF data file
    check_file = root + "10_seconds/BIMFs/all_BIMFs_post_EC.pickle"
    if not os.path.exists(check_file):
        combine_BIMF_dfs()
