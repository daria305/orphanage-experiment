import numpy as np
from matplotlib import pyplot as plt

from read_data import read_data
from group_data import add_median_column, add_max_column, find_the_best_orphanage, assign_q_values, setup_rolling_count, \
    group_tips_by_q, group_times_by_q, exclude_columns, filter_by_qs
from plot_data import plot_tips_by_node, plot_cumulative_orphanage_by_time, plot_grafana_tips_q_for_all_k, \
    plot_grafana_times_q_for_all_k, plot_tips_final_times

timeCol = "Time"
advCol = "Tips adversary:9311"

DATA_PATH_K2 = "data/orphanage/final/k_2"
DATA_PATH_K4 = "data/orphanage/final/k_4"
DATA_PATH_K8 = "data/orphanage/final/k_8"

Ks = [2, 4, 8]

K2_Qs = [0.5, 0.53, 0.55]
K4_Qs = [0.65, 0.7, 0.75, 0.8]
K8_Qs = [0.65, 0.7, 0.75, 0.8, 0.88, 0.93]


def analyse_tips(df):
    plot_tips_by_node(df)


def the_best_orphanage_start_and_stop_points(orphanage_df):
    # K = 2
    qs = [0.5, 0.53, 0.55]
    for q in qs:
        interval, o = find_the_best_orphanage(orphanage_df, q, 50, 5)
        print("Q:,", q, "The best orphanage", o, "found for interval:", interval)


def orphanage_by_time():
    _, _, _, _, orphanage_df_k2 = read_data(DATA_PATH_K2)
    _, _, _, _, orphanage_df_k4 = read_data(DATA_PATH_K4)
    _, _, _, _, orphanage_df_k8 = read_data(DATA_PATH_K8)

    plot_cumulative_orphanage_by_time(orphanage_df_k2, K2_Qs, "orphanage_k2")
    plot_cumulative_orphanage_by_time(orphanage_df_k4, K4_Qs, "orphanage_k4")
    plot_cumulative_orphanage_by_time(orphanage_df_k8, K8_Qs, "orphanage_k8")


def tips_per_q():
    paths = [DATA_PATH_K2, DATA_PATH_K4, DATA_PATH_K8]
    tips_dfs, conf_dfs = [], []
    for i, k in enumerate(Ks):
        setup_rolling_count()
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(paths[i])
        print(k)
        tips_df, mpsi_df, conf_df = assign_q_values(tips_df, mpsi_df, conf_df, orphanage_df)
        tips_df = add_median_column(tips_df)
        conf_df = add_max_column(conf_df)
        conf_df["Max"] = conf_df["Max"] / float(1000000000 * 60)  # convert to seconds
        # group by q for the plots
        tips_by_q_df = group_tips_by_q(tips_df)
        conf_by_q_df = group_times_by_q(conf_df)

        tips_dfs.append(tips_by_q_df)
        conf_dfs.append(conf_by_q_df)

    plot_grafana_tips_q_for_all_k(Ks, tips_dfs)
    plot_grafana_times_q_for_all_k(Ks, conf_dfs)


def grafana_like_plots():
    paths = [DATA_PATH_K2, DATA_PATH_K4, DATA_PATH_K8]

    q_for_k = {
        2: 0.35,
        4: 0.5,
        8: 0.6
    }

    for i, k in enumerate(Ks):
        setup_rolling_count()
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(paths[i])
        tips_df, mpsi_df, conf_df = assign_q_values(tips_df, mpsi_df, conf_df, orphanage_df)

        # get data index for exp==1 => first q value to filter out all q=0 before
        min_index = tips_df[round(tips_df['q'], 2) == round(q_for_k[k], 2)].index.min()
        tips_df = tips_df[tips_df.index >= min_index]

        min_index = conf_df[round(conf_df['q'], 2) == round(q_for_k[k], 2)].index.min()
        conf_df = conf_df[conf_df.index >= min_index]

        # tips_df = filter_by_qs(tips_df, q_for_k[k])
        # conf_df = filter_by_qs(conf_df, q_for_k[k])

        plot_tips_final_times(tips_df, conf_df, k)


if __name__ == "__main__":
    # tips_df = add_stat_columns(tips_df)

    # the_best_orphanage_start_and_stop_points(orphanage_df)

    orphanage_by_time()
    tips_per_q()

    grafana_like_plots()
