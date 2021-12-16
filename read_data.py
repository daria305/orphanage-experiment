import datetime
import time

import pandas as pd
import os

startPath = "data/orphanage/30mpsDur8"



def read_data(path_to_data):

    file_names = [files for _, _, files in os.walk(path_to_data)]
    for file in file_names[0]:
        if file == "" or file[-4:] == ".csv":
            if file.startswith('Message Per Second Per Issuer'):
                mpsi_file = os.path.join(path_to_data, file)
                mpsi_df = read_grafana_file(mpsi_file, "mps")

            if file.startswith('Message Per Second Per Type'):
                mpst_file = os.path.join(path_to_data, file)
                mpst_df = read_grafana_file(mpst_file, "mps")

            if file.startswith('Tips'):
                tips_file = os.path.join(path_to_data, file)
                tips_df = read_grafana_file(tips_file, "num")

            if file.startswith('Message Finalization'):
                finalization_file = os.path.join(path_to_data, file)
                finalization_df = read_grafana_file(finalization_file, "times")

            if file.startswith('orphanage'):
                orphanage_file = os.path.join(path_to_data, file)
                orphanage_df = read_orphanage_file(orphanage_file)

    return mpsi_df, mpst_df, tips_df, finalization_df, orphanage_df


def read_grafana_file(file, type):
    with open(file) as rf:
        df = pd.read_csv(rf, sep=',')
    df.dropna(how='any', thresh=2, axis=0, inplace=True)
    if type == "mps":
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(parse_mps)
    elif type == "times":
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(parse_times)
    elif type == "num":
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(parse_num)

    return df


def read_orphanage_file(file):
    results_col = ["expId", "q", "mps", "honestOrphanageRate", "advOrphanageRate", "totalOrphans", "honestOrphans",
                   "advOrphans", "totalIssued", "honestIssued", "advIssued", "requester", "attackDuration",
                   "intervalNum", "intervalStart", "intervalStop"]
    with open(file) as rf:
        df = pd.read_csv(rf, sep=',')
    df.columns = results_col

    return df


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
    DATA_PATH = "data/orphanage/equalSnapshot/30mpsDur8"
    mpsi_df, mpst_df, tips_df, finalization_df, orphanage_df = read_data(DATA_PATH)
    print(mpsi_df.head())
    print(mpst_df.head())
    print(mpst_df.head())
    print(mpst_df.head())
    print(tips_df.head())
    print(finalization_df.head())
    print(orphanage_df.head())
