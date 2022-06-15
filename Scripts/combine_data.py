import os
import pickle
from sys import platform

import pandas as pd
from omegaconf import OmegaConf

# Get noisy rec list from the configuration file
config = OmegaConf.load("config.yaml")
noisy_recs = config.noisy_recs

# Get root folder based on which operating system I'm working on
root = "Data/tDCS_EEG_data/Epochs/"
if platform == "darwin":
    root = "/Users/erlahrafnkelsdottir/Documents/DepEEG/" + root

# Lists of ready files
recs116s = os.listdir(root + "116_seconds/")
recs116s = sorted(list(set(recs116s) - set(noisy_recs)))
recs116s_EC_pre = [r for r in recs116s if ("EC" in r) and ("pre" in r)]
recs116s_EO_pre = [r for r in recs116s if ("EO" in r) and ("pre" in r)]
recs116s_EC_post = [r for r in recs116s if ("EC" in r) and ("post" in r)]
recs116s_EO_post = [r for r in recs116s if ("EO" in r) and ("post" in r)]


def get_subject_id(filename: str) -> int:
    name_split = filename.split("_")
    subject = name_split[0][1:]
    id = int(subject)
    return id


def create_combined_dfs():
    # Let's create some combined dataframes
    # -------------- EC PRE combined dataframe --------------

    # Initialize
    first_EC_pre = recs116s_EC_pre[0]
    first_EC_pre_id = get_subject_id(first_EC_pre)
    first_EC_pre_dep = 1 if "D" in first_EC_pre else 0
    all_pre_EC_116s = pd.read_csv(
        root + "116_seconds/" + first_EC_pre, sep="\t", index_col=False
    )
    all_pre_EC_116s["Subject_ID"] = first_EC_pre_id
    all_pre_EC_116s["Depressed"] = first_EC_pre_dep

    # Then append all other records with the same categorization
    for file in recs116s_EC_pre:
        if file == recs116s_EC_pre[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_pre_EC_116s = all_pre_EC_116s.append(record)

    # Save as .txt
    all_pre_EC_116s.to_csv(
        root + "116_seconds/All_pre_EC_116s.txt", sep="\t", index=False
    )

    # Save as .pickle
    with open(root + "116_seconds" + "/all_pre_EC_116s.pickle", "wb") as f:
        pickle.dump(all_pre_EC_116s, f)

    print("EC PRE combined dataframe SAVED.")

    # -------------- EO PRE combined dataframe --------------

    # Initialize
    first_EO_pre = recs116s_EO_pre[0]
    first_EO_pre_id = get_subject_id(first_EO_pre)
    first_EO_pre_dep = 1 if "D" in first_EO_pre else 0
    all_pre_EO_116s = pd.read_csv(
        root + "116_seconds/" + first_EO_pre, sep="\t", index_col=False
    )
    all_pre_EO_116s["Subject_ID"] = first_EO_pre_id
    all_pre_EO_116s["Depressed"] = first_EO_pre_dep

    # Then append all other records with the same categorization
    for file in recs116s_EO_pre:
        if file == recs116s_EO_pre[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_pre_EO_116s = all_pre_EO_116s.append(record)

    # Save as .txt
    all_pre_EO_116s.to_csv(
        root + "116_seconds/All_pre_EO_116s.txt", sep="\t", index=False
    )

    # Save as .pickle
    with open(root + "116_seconds" + "/all_pre_EO_116s.pickle", "wb") as f:
        pickle.dump(all_pre_EO_116s, f)

    print("EO PRE combined dataframe SAVED.")

    # -------------- EC POST combined dataframe --------------

    # Initialize
    first_EC_post = recs116s_EC_post[0]
    first_EC_post_id = get_subject_id(first_EC_post)
    first_EC_post_dep = 1 if "D" in first_EC_post else 0
    all_post_EC_116s = pd.read_csv(
        root + "116_seconds/" + first_EC_post, sep="\t", index_col=False
    )
    all_post_EC_116s["Subject_ID"] = first_EC_post_id
    all_post_EC_116s["Depressed"] = first_EC_post_dep

    # Then append all other records with the same categorization
    for file in recs116s_EC_post:
        if file == recs116s_EC_post[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_post_EC_116s = all_post_EC_116s.append(record)

    # Save as .txt
    all_post_EC_116s.to_csv(
        root + "116_seconds/All_post_EC_116s.txt", sep="\t", index=False
    )

    # Save as .pickle
    with open(root + "116_seconds" + "/all_post_EC_116s.pickle", "wb") as f:
        pickle.dump(all_post_EC_116s, f)

    print("EC POST combined dataframe SAVED.")

    # -------------- EO POST combined dataframe --------------

    # Initialize
    first_EO_post = recs116s_EO_post[0]
    first_EO_post_id = get_subject_id(first_EO_post)
    first_EO_post_dep = 1 if "D" in first_EO_post else 0
    all_post_EO_116s = pd.read_csv(
        root + "116_seconds/" + first_EO_post, sep="\t", index_col=False
    )
    all_post_EO_116s["Subject_ID"] = first_EO_post_id
    all_post_EO_116s["Depressed"] = first_EO_post_dep

    # Then append all other records with the same categorization
    for file in recs116s_EO_post:
        if file == recs116s_EO_post[0]:
            continue
        id = get_subject_id(file)
        dep = 1 if "D" in file else 0
        record = pd.read_csv(root + "116_seconds/" + file, sep="\t", index_col=False)
        record["Subject_ID"] = id
        record["Depressed"] = dep
        all_post_EO_116s = all_post_EO_116s.append(record)

    # Save as .txt
    all_post_EO_116s.to_csv(
        root + "116_seconds/All_post_EO_116s.txt", sep="\t", index=False
    )

    # Save as .pickle
    with open(root + "116_seconds" + "/all_post_EO_116s.pickle", "wb") as f:
        pickle.dump(all_post_EO_116s, f)

    print("EO POST combined dataframe SAVED.")

    # -------------- EVERYTHING IN ONE DATAFRAME --------------

    # Add pre/post and EO/EC columns
    all_pre_EC_116s["Post"] = 0
    all_pre_EC_116s["EO"] = 0
    all_pre_EO_116s["Post"] = 0
    all_pre_EO_116s["EO"] = 1
    all_post_EC_116s["Post"] = 1
    all_post_EC_116s["EO"] = 0
    all_post_EC_116s["Post"] = 1
    all_post_EC_116s["EO"] = 1

    # Combine
    all_116s = all_post_EC_116s
    all_116s = all_116s.append(all_post_EO_116s)
    all_116s = all_116s.append(all_pre_EC_116s)
    all_116s = all_116s.append(all_pre_EO_116s)

    # Save as .txt
    all_116s.to_csv(root + "116_seconds/All_116s.txt", sep="\t", index=False)
    print("ALL DATA combined dataframe SAVED.")

    # Save as .pickle
    with open(root + "116_seconds" + "/all_116s.pickle", "wb") as f:
        pickle.dump(all_116s, f)

    return


if __name__ == "__main__":
    # Check whether we have already made and saved the combined data files
    check_file = root + "116_seconds/All_116s.txt"
    if not os.path.exists(check_file):
        create_combined_dfs()
