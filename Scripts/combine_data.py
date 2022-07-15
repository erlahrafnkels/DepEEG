import os
import pickle
from sys import platform

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import zscore

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


def combine_BIMF_dfs(name):
    # Get file names
    path = root + "10_seconds/BIMFs/"
    bimf_files = os.listdir(path)
    bimf_files = [f for f in bimf_files if name in f]

    # Initialize dataframe
    with open(path + bimf_files[0], "rb") as f:
        first_file = pickle.load(f)
    all_BIMFs_df = first_file[first_file.columns[0:5]]
    bimf_files = bimf_files[1:]
    print("Added", bimf_files[0])

    # Concatenate all other files
    for file in bimf_files:
        with open(path + file, "rb") as f:
            current_file = pickle.load(f)
        bimf_cols = current_file.columns[0:5]
        current_file = current_file[bimf_cols]
        all_BIMFs_df = pd.concat([all_BIMFs_df, current_file], axis=1)
        print("Added", file)

    # Add ID columns
    id_cols = first_file[first_file.columns[-2:]]
    all_BIMFs_df = pd.concat([all_BIMFs_df, id_cols], axis=1)
    all_BIMFs_df["Depressed"] = all_BIMFs_df.apply(
        lambda row: 1 if row["Subject_ID"] in depressed_ids else 0, axis=1
    )

    # Save as .pickle
    with open(path + "/all_BIMFs_" + name + ".pickle", "wb") as f:
        pickle.dump(all_BIMFs_df, f)

    print("Combined BIMF dataframe SAVED.")

    return


def combine_feature_dfs(non_vmd_name, vmd_name, which, z_norm):
    path = "Data/tDCS_EEG_data/Features/"

    # Get non-VMD feature table
    with open(path + non_vmd_name, "rb") as f:
        non_vmd_df = pickle.load(f)

    # Get VMD feature table
    with open(path + vmd_name, "rb") as f:
        vmd_df = pickle.load(f)

    # Combine
    all_features_df = pd.merge(
        non_vmd_df,
        vmd_df,
        left_on=["Subject_ID", "Depression"],
        right_on=["Subject_ID", "Depression"],
    )

    # Move ID columns to the end
    sub = all_features_df.pop("Subject_ID")
    dep = all_features_df.pop("Depression")
    all_features_df["Subject_ID"] = sub
    all_features_df["Depression"] = dep

    # Save as .pickle
    with open(path + "ALL_feature_df_" + which + ".pickle", "wb") as f:
        pickle.dump(all_features_df, f)

    print(f"Combined feature dataframe for {which} SAVED.")

    return


def split_other_features(which):
    # Split feature tables into subtables by feature set
    # We have the VMD and all combined tables already
    # Need subtables for statistical, pow per channel and pow per region
    stat_names = ["Mean", "Var", "Skew", "Kurt", "Subject_ID", "Depression"]
    region_names = [
        "frontal",
        "temporal",
        "parietal",
        "occipital",
        "central",
        "Subject_ID",
        "Depression",
    ]
    feature_path = "Data/tDCS_EEG_data/Features/"

    # Get "other" feature table
    with open(feature_path + "other_feature_df_" + which + ".pickle", "rb") as f:
        other_features_df = pickle.load(f)

    # Statistical
    stat_df = other_features_df.copy()
    stat_df = stat_df.filter(regex="|".join(stat_names))
    print("Stats:\t\t", stat_df.shape)

    # Power per region
    pow_reg_df = other_features_df.copy()
    pow_reg_df = pow_reg_df.filter(regex="|".join(region_names))
    print("Region powers:\t", pow_reg_df.shape)

    # Power per channel
    all_cols = list(other_features_df.columns)
    remove_cols = list(pow_reg_df.columns[:-2]) + list(stat_df.columns[:-2])
    pow_chan_names = [col for col in all_cols if col not in remove_cols]
    pow_chan_df = other_features_df.copy()
    pow_chan_df = pow_chan_df.filter(regex="|".join(pow_chan_names))
    print("Channel powers:\t", pow_chan_df.shape)

    # Save subdataframes as .pickle
    with open(feature_path + "stat_feature_df_" + which + ".pickle", "wb") as f:
        pickle.dump(stat_df, f)

    with open(feature_path + "pow_reg_feature_df_" + which + ".pickle", "wb") as f:
        pickle.dump(pow_reg_df, f)

    with open(feature_path + "pow_chan_feature_df_" + which + ".pickle", "wb") as f:
        pickle.dump(pow_chan_df, f)

    print(f"Split feature dataframes for {which} SAVED.")


def combine_norm_features():
    path = "Data/tDCS_EEG_data/Features/"

    # Get non-VMD feature table PRE
    with open(path + "non-norm-other_feature_df_all_pre_EC.pickle", "rb") as f:
        non_vmd_df_pre = pickle.load(f)

    # Get VMD feature table PRE
    with open(path + "non-norm-vmd_feature_df_all_pre_EC.pickle", "rb") as f:
        vmd_df_pre = pickle.load(f)

    # Get non-VMD feature table POST
    with open(path + "non-norm-other_feature_df_all_post_EC.pickle", "rb") as f:
        non_vmd_df_post = pickle.load(f)

    # Get VMD feature table POST
    with open(path + "non-norm-vmd_feature_df_all_post_EC.pickle", "rb") as f:
        vmd_df_post = pickle.load(f)

    # Add pre/post columns
    non_vmd_df_pre["Post"] = 0
    vmd_df_pre["Post"] = 0
    non_vmd_df_post["Post"] = 1
    vmd_df_post["Post"] = 1

    print(non_vmd_df_pre.shape)
    print(vmd_df_pre.shape)
    print(non_vmd_df_post.shape)
    print(vmd_df_post.shape)

    # Combine PRES
    all_features_pre_df = pd.merge(
        non_vmd_df_pre,
        vmd_df_pre,
        left_on=["Subject_ID", "Depression", "Post"],
        right_on=["Subject_ID", "Depression", "Post"],
    )

    # Combine POSTS
    all_features_post_df = pd.merge(
        non_vmd_df_post,
        vmd_df_post,
        left_on=["Subject_ID", "Depression", "Post"],
        right_on=["Subject_ID", "Depression", "Post"],
    )

    # Combine ALL
    all_features_df = all_features_pre_df.append(
        all_features_post_df, ignore_index=True
    )

    print(all_features_df.shape)

    # Remove ID columns for normalization
    sub = all_features_df.pop("Subject_ID")
    dep = all_features_df.pop("Depression")
    post = all_features_df.pop("Post")

    print(all_features_df.shape)

    all_features_df_cols = all_features_df.columns

    # When feature matrix has been filled with values, we normalize it
    feature_mat = np.array(all_features_df).astype(float)
    feature_mat = zscore(feature_mat, axis=None)  # All table
    # feature_mat = zscore(feature_mat, axis=1)  # Per column

    # Then put it into a dataframe so we have the column names
    all_features_df = pd.DataFrame(feature_mat, columns=all_features_df_cols)
    all_features_df["Subject_ID"] = sub
    all_features_df["Depression"] = dep
    all_features_df["Post"] = post

    print(all_features_df.shape)

    # Save as .pickle
    with open(path + "ALL_feature_df_EC.pickle", "wb") as f:
        pickle.dump(all_features_df, f)

    print("Combined feature dataframe SAVED.")

    return


if __name__ == "__main__":
    # What to run
    create_combined_ppee_dfs = False
    create_combined_BIMF_dfs = False
    create_combined_feature_dfs = False
    create_split_other_features = False
    create_combined_norm_features = False
    which = "all_pre_EO"

    if create_combined_ppee_dfs:
        create_combined_dfs()

    if create_combined_BIMF_dfs:
        name = which[4:]
        combine_BIMF_dfs(name)

    if create_combined_feature_dfs:
        non_vmd_df = "other_feature_df_" + which + ".pickle"
        vmd_df = "vmd_feature_df_" + which + ".pickle"
        combine_feature_dfs(non_vmd_df, vmd_df, which, z_norm=True)

    if create_split_other_features:
        split_other_features(which)

    if create_combined_norm_features:
        combine_norm_features()
