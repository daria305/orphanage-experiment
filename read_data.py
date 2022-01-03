import datetime
import time

import pandas as pd
import os


def get_file_paths(path_to_data):
    file_names = []
    file_paths = []
    for paths, _, files in os.walk(path_to_data):
        if len(files) > 0:
            file_names.extend(files)
            file_paths.extend([paths] * len(files))
    path_file = zip(file_paths, file_names)
    sorted_tuples = sorted(path_file, key=sort_by_sub_dir)
    lists = list(zip(*sorted_tuples))
    file_paths = lists[0]
    file_names = lists[1]
    return file_paths, file_names


def sort_by_sub_dir(zipped):
    path, _ = zipped
    catalog_num = int(path.split("/")[-1])
    return catalog_num


def read_data(path_to_data):
    file_paths, file_names = get_file_paths(path_to_data)

    mpsi_file = []
    mpst_file = []
    tips_file = []
    finalization_file = []
    orphanage_file = []

    for file, path in zip(file_names, file_paths):
        if file == "" or file[-4:] == ".csv":
            if file.startswith('Message Per Second Per Issuer'):
                mpsi_file.append(os.path.join(path, file))

            if file.startswith('Message Per Second Per Type'):
                mpst_file.append(os.path.join(path, file))

            if file.startswith('Tips'):
                tips_file.append(os.path.join(path, file))

            if file.startswith('Message Finalization'):
                finalization_file.append(os.path.join(path, file))

            if file.startswith('orphanage'):
                orphanage_file.append(os.path.join(path, file))
    # read the data to pandas
    tips_df = save_grafana_files_to_df(tips_file, "num")
    mpst_df = save_grafana_files_to_df(mpst_file, "mps")
    mpsi_df = save_grafana_files_to_df(mpsi_file, "mps")
    finalization_df = save_grafana_files_to_df(finalization_file, "times")

    orphanage_df = save_orphanage_files_to_df(orphanage_file)

    return mpsi_df, mpst_df, tips_df, finalization_df, orphanage_df


def save_grafana_files_to_df(files, data_type):
    df = None
    for file in files:
        df = read_grafana_file(file, data_type, df)
    return df


def save_orphanage_files_to_df(files):
    df = None
    for file in files:
        df = read_orphanage_file(file, df)
    return df


def read_grafana_file(file, type, existing_df):
    with open(file) as rf:
        df = pd.read_csv(rf, sep=',')
    if type == "mps":
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(parse_mps)
    elif type == "times":
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(parse_times)
    elif type == "num":
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(parse_num)
    df["Time"] = pd.to_datetime(df["Time"])

    if existing_df is not None:
        existing_df = existing_df.append(df, ignore_index=True)
    else:
        existing_df = df

    return existing_df


def read_orphanage_file(file, existing_df):
    results_col = ["expId", "q", "mps", "honestOrphanageRate", "advOrphanageRate", "totalOrphans", "honestOrphans",
                   "advOrphans", "totalIssued", "honestIssued", "advIssued", "requester", "attackDuration",
                   "intervalNum", "intervalStart", "intervalStop"]
    with open(file) as rf:
        df = pd.read_csv(rf, sep=',')
    df["intervalStart"] = pd.to_datetime(df["intervalStart"], unit='us')
    df["intervalStop"] = pd.to_datetime(df["intervalStop"], unit='us')
    if existing_df is None:
        df.columns = results_col
        existing_df = df
    else:
        existing_df = existing_df.append(df, ignore_index=True)
    return existing_df


def parse_mps(cell):
    a = cell.values
    for i, val in enumerate(a):
        if pd.notna(val):
            a[i] = float(val.split(" ")[0])
    return cell


# time unit in df ms
def parse_times(cell):
    values = cell.values
    mic = 1000
    ms = mic * 1000
    sec = ms * 1000
    minute = sec * 60
    h = minute * 60

    for i, val in enumerate(values):
        if pd.notna(val):
            s = val.split(" ")
            if s[1] == 's':
                t = int(float(s[0]) * sec)
            elif s[1] == 'ms':
                t = int(float(s[0]) * ms)
            elif s[1] == 'min':
                t = int(float(s[0]) * minute)
            else:
                raise Exception("missing time unit conversion rule")
            values[i] = t
    return cell


def parse_num(cell):
    a = cell.values
    for i, val in enumerate(a):
        if pd.notna(val):
            a[i] = int(val)
    return cell


if __name__ == "__main__":
    DATA_PATH = "data/orphanage/final/k_4"
    mpsi_df, mpst_df, tips_df, finalization_df, orphanage_df = read_data(DATA_PATH)
    print(mpsi_df.head())
    print(mpst_df.head())
    print(mpst_df.head())
    print(mpst_df.head())
    print(tips_df.head())
    print(finalization_df.head())
    print(orphanage_df.head())
    get_file_paths(DATA_PATH)
