import pandas as pd

from read_data import read_data
from group_data import add_median_column, add_max_column, find_the_best_orphanage, \
    group_tips_by_q, group_times_by_q, add_avg_column, \
    assign_q_based_on_adv_rate, get_all_qs, merge_nodes_data_with_max, exclude_columns, ADV_FINALIZATION_COL
from plot_data import plot_tips_by_node, plot_cumulative_orphanage_by_time, plot_grafana_tips_q_for_all_k, \
    plot_grafana_times_q_for_all_k, plot_tips_final_times, plot_tips_infinite, plot_times_infinite

DATA_PATH = "data"


def data_path_orphanage(k, subdir):
    return "{}/k_{}/orphanage/{}".format(DATA_PATH, k, subdir)


def data_path_infinite(k, subdir):
    return "{}/k_{}/tippool/{}".format(DATA_PATH, k, subdir)


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


def analyse_tips(df: pd.DataFrame):
    plot_tips_by_node(df)


def the_best_orphanage_start_and_stop_points(orphanage_df: pd.DataFrame):
    qs = [0.5, 0.53, 0.55]
    for q in qs:
        interval, o = find_the_best_orphanage(orphanage_df, q, 50, 5)
        print("Q:,", q, "The best orphanage", o, "found for interval:", interval)


def orphanage_by_time():
    for k in Ks:
        _, _, _, _, orphanage_df = read_data(data_path_orphanage(k, ""))
        plot_cumulative_orphanage_by_time(orphanage_df, K_Q[k], "orphanage_k{}".format(k))


def tips_per_q():
    paths = ["{}/k_{}/orphanage".format(DATA_PATH, k) for k in Ks]
    tips_dfs, conf_dfs = [], []

    for i, k in enumerate(Ks):
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(paths[i])
        mpsi_df, tips_df, conf_df = assign_q_based_on_adv_rate(mpsi_df, tips_df, conf_df)
        tips_df = add_median_column(tips_df)
        conf_df = add_median_column(conf_df)
        # group by q for the plots
        tips_by_q_df = group_tips_by_q(tips_df)
        conf_by_q_df = group_times_by_q(conf_df)

        tips_dfs.append(tips_by_q_df)
        conf_dfs.append(conf_by_q_df)

    plot_grafana_tips_q_for_all_k(Ks, tips_dfs)
    plot_grafana_times_q_for_all_k(Ks, conf_dfs)


def grafana_like_plots():
    paths = ["{}/k_{}/orphanage/".format(DATA_PATH, k) for k in Ks]
    mpsi, tips, conf, qs = [], [], [], []
    max_tip = 0
    max_conf = 0

    for i, k in enumerate(Ks):
        mpsi_df, _, tips_df, conf_df, _ = read_data(paths[i])
        mpsi_df, tips_df, conf_df = assign_q_based_on_adv_rate(mpsi_df, tips_df, conf_df)
        mpsi_df, tips_df = merge_nodes_data_with_max(mpsi_df, tips_df)
        tips.append(tips_df)
        conf.append(conf_df)

        conf_df_max = exclude_columns(conf_df, ['q', 'Time', ADV_FINALIZATION_COL]).max(axis=0).max()

        # find max value to limit y-axis
        max_tip = max(tips_df['Max Tips'].max(), max_tip)
        max_conf = max(conf_df_max, max_conf)

    for i, k in enumerate(Ks):
        plot_tips_final_times(tips[i], conf[i], k, max_tip, max_conf)


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
        2: [0.5],
        4: [0.5, 0.75],
        8: [0.75, 0.88],
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
    # orphanage_by_time()
    # tips_per_q()
    grafana_like_plots()
    # infinite_tips_plots() TODO make it more readable
    # infinite_times_plots()
