import os.path

import numpy as np
from matplotlib import pyplot as plt

from read_data import read_data, get_file_paths
from group_data import add_median_column, add_max_column, find_the_best_orphanage, assign_q_values, \
    group_tips_by_q, group_times_by_q, exclude_columns, filter_by_qs, add_avg_column, \
    assign_q_based_on_adv_rate
from plot_data import plot_tips_by_node, plot_cumulative_orphanage_by_time, plot_grafana_tips_q_for_all_k, \
    plot_grafana_times_q_for_all_k, plot_tips_final_times, plot_tips_infinite, plot_times_infinite

timeCol = "Time"
advCol = "Tips adversary:9311"

DATA_PATH = "data/"


def data_path_orphanage(k, subdir):
    return "{}/k_{}/orphanage/{}".format(DATA_PATH, k, subdir)


def data_path_infinite(k, subdir):
    return "{}/k_{}/tippool/{}".format(DATA_PATH, k, subdir)


def read_sub_dirs():
    file_paths, _ = get_file_paths(DATA_PATH)
    for path in file_paths:
        s = path.split("/")
        k = int(s[0].split("_")[1])
        sub_dir = s[2]
        SUB_DIRS[k] = SUB_DIRS[k].append(sub_dir)


Ks = [2, 4, 8, 16]
SUB_DIRS = {k: [] for k in Ks}

K_Q = {
    2: [0.5, 0.53, 0.55],
    4: [0.7, 0.75, 0.8],
    8: [0.8, 0.88, 0.93],
    16: [0.9, 0.94, 0.99]
}

K_CRITICAL = {
    2: 0.5,
    4: 0.75,
    8: 0.88,
    16: 0.93
}


def analyse_tips(df):
    plot_tips_by_node(df)


def the_best_orphanage_start_and_stop_points(orphanage_df):
    # K = 2
    qs = [0.5, 0.53, 0.55]
    for q in qs:
        interval, o = find_the_best_orphanage(orphanage_df, q, 50, 5)
        print("Q:,", q, "The best orphanage", o, "found for interval:", interval)


def orphanage_by_time():
    for k in Ks:
        _, _, _, _, orphanage_df = read_data(data_path_orphanage(k, ""))
        plot_cumulative_orphanage_by_time(orphanage_df, K_Q[k], "orphanage_k{}".format(k))


def tips_per_q():
    paths = ["{}/k_{}".format(DATA_PATH, k) for k in Ks]
    tips_dfs, conf_dfs = [], []

    for i, k in enumerate(Ks):
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(paths[i])
        mpsi_df, tips_df, conf_df = assign_q_based_on_adv_rate(mpsi_df, tips_df, conf_df)

        tips_df = add_median_column(tips_df)
        conf_df = add_max_column(conf_df)
        conf_df["Max"] = conf_df["Max"]  # convert to seconds
        # group by q for the plots
        tips_by_q_df = group_tips_by_q(tips_df)
        conf_by_q_df = group_times_by_q(conf_df)

        tips_dfs.append(tips_by_q_df)
        conf_dfs.append(conf_by_q_df)

    plot_grafana_tips_q_for_all_k(Ks, tips_dfs)
    plot_grafana_times_q_for_all_k(Ks, conf_dfs)


def grafana_like_plots():
    paths = ["{}/k_{}".format(DATA_PATH, k) for k in Ks]

    for i, k in enumerate(Ks):
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(paths[i])
        # tips_df, mpsi_df, conf_df = assign_q_values(tips_df, mpsi_df, conf_df, orphanage_df)
        mpsi_df, tips_df, conf_df = assign_q_based_on_adv_rate(mpsi_df, tips_df, conf_df)

        # get data index for exp==1 => first q value to filter out all q=0 before
        a = round(tips_df['q'], 2)
        b = round(K_CRITICAL[k], 2)
        # min_index = tips_df[a == b].index.min()
        # tips_df = tips_df[tips_df.index >= min_index]
        #
        # min_index = conf_df[round(conf_df['q'], 2) == round(K_CRITICAL[k], 2)].index.min()
        # conf_df = conf_df[conf_df.index >= min_index]

        # tips_df = filter_by_qs(tips_df, q_for_k[k])
        # conf_df = filter_by_qs(conf_df, q_for_k[k])

        plot_tips_final_times(tips_df, conf_df, k)


def infinite_tips_plots():
    qs = [0.5, 0.5, 0.75, 0.75, 0.88, 0.88, 0.93]
    ks = [2, 4, 4, 8, 8, 16, 16]
    sub_dirs = [0, 0, 1, 0, 1, 0, 1]
    tips_dfs = []
    for i, k in enumerate(ks):
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(data_path_infinite(k, sub_dirs[i]))
        tips_df = add_avg_column(tips_df)
        tips_dfs.append(tips_df)
    plot_tips_infinite(tips_dfs, ks, qs)


def infinite_times_plots():
    ks = [2, 4, 8, 16]
    q_for_k = {
        2:  [0.5],
        4:  [0.5, 0.75],
        8:  [0.75, 0.88],
        16: [0.88, 0.93]
    }

    for i, k in enumerate(ks):
        conf_dfs = []
        for j, q in enumerate(q_for_k[k]):
            path = data_path_infinite(k, j)
            mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(path)
            conf_dfs.append(conf_df)
        plot_times_infinite(conf_dfs, k, q_for_k[k])


if __name__ == "__main__":
    # tips_df = add_stat_columns(tips_df)

    # the_best_orphanage_start_and_stop_points(orphanage_df)
    orphanage_by_time()
    tips_per_q()
    grafana_like_plots()
    infinite_tips_plots()
    infinite_times_plots()
