import pandas as pd

from read_data import read_data
from group_data import add_median_column, add_max_column, find_the_best_orphanage, \
    group_tips_by_q, group_times_by_q, add_avg_column, \
    assign_q_based_on_adv_rate, get_all_qs, merge_nodes_data_with_max, exclude_columns, ADV_FINALIZATION_COL, \
    add_moving_avg_column, idle_spam_time_end, cut_by_time, filter_by_q, filter_beginning_tips
from plot_data import plot_tips_by_node, plot_cumulative_orphanage_by_time, plot_grafana_tips_q_for_all_k, \
    plot_grafana_times_q_for_all_k, plot_tips_final_times, plot_tips_infinite, plot_times_infinite, plot_maxage_tips, \
    plot_maxage_conf, plot_cumulative_orphanage_maxage_by_time, plot_tips_closer_look, plot_tips_final_times_summary, \
    plot_infinite_summary

DATA_PATH = "data"


def data_path_orphanage(k, subdir):
    return "{}/k_{}/orphanage/{}".format(DATA_PATH, k, subdir)


def data_path_maxage(k, subdir):
    return "{}/k_{}/maxage/{}".format(DATA_PATH, k, subdir)


def data_path_infinite(k, subdir):
    return "{}/k_{}/tippool/{}".format(DATA_PATH, k, subdir)


def data_path_maxage(k, subdir):
    return "{}/k_{}/maxage/{}".format(DATA_PATH, k, subdir)


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


def infinite_tips_plots2():
    qs = [0.5, 0.5, 0.75, 0.75, 0.88, 0.88, 0.93]
    ks = [2, 4, 4, 8, 8, 16, 16]
    sub_dirs = [0, 0, 1, 0, 1, 0, 1]
    tips_dfs = []
    max_tip = 0
    for i, k in enumerate(ks):
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(data_path_infinite(k, sub_dirs[i]))
        tips_df = add_avg_column(tips_df)
        tips_dfs.append(tips_df)
        max_tip = max(tips_df["Avg"].max(), max_tip)
    # plot_tips_infinite(tips_dfs, ks, qs)


def infinite_times_plots():
    ks = [2, 4, 8, 16]
    q_for_k = {
        2: [0.5],
        4: [0.5, 0.75],
        8: [0.75, 0.88],
        16: [0.88, 0.93]
    }
    conf = []
    max_conf = 0
    for i, k in enumerate(ks):
        conf_dfs = []
        for j, q in enumerate(q_for_k[k]):
            path = data_path_infinite(k, j)
            _, _, _, conf_df, _ = read_data(path)
            conf_df_max = exclude_columns(conf_df, ['Time', ADV_FINALIZATION_COL]).max(axis=0).max()
            max_conf = max(max_conf, conf_df_max)
            conf_dfs.append(conf_df)

        conf.append(conf_dfs)
    for i, k in enumerate(ks):
        plot_times_infinite(conf[i], k, q_for_k[k], max_conf)


def infinite_tips_plots():
    ks = [2, 4, 8, 16]
    q_for_k = {
        2: [0.5],
        4: [0.5, 0.75],
        8: [0.75, 0.88],
        16: [0.88, 0.93]
    }
    tips = []
    max_tip = 0
    for i, k in enumerate(ks):
        tips_dfs = []
        for j, q in enumerate(q_for_k[k]):
            path = data_path_infinite(k, j)
            _, _, tips_df, _, _ = read_data(path)
            tips_df = add_avg_column(tips_df)
            max_tip = max(tips_df["Avg"].max(), max_tip)
            tips_dfs.append(tips_df)

        tips.append(tips_dfs)
    for i, k in enumerate(ks):
        plot_tips_infinite(tips[i], k, q_for_k[k], max_tip)


def max_age_plots():
    k = 2
    max_age = [20, 40, 60, 80, 100, 120, 180, 300]
    tips = []
    confs = []
    orphanages = []
    for i, age in enumerate(max_age):
        path = data_path_maxage(2, i)
        mpsi, _, tips_df, conf_df, orphanage_df = read_data(path)
        start_time = idle_spam_time_end(mpsi)
        tips_df = cut_by_time(tips_df, start_time)
        conf_df = cut_by_time(conf_df, start_time)

        tips_df = add_median_column(tips_df)
        conf_df2 = add_moving_avg_column(conf_df, 10)
        # conf_df = add_max_column(conf_df)

        tips.append(tips_df)
        confs.append(conf_df2)
        orphanages.append(orphanage_df)

    plot_maxage_tips(max_age, tips)
    plot_maxage_conf(max_age, confs)
    # plot_maxage_conf()
    # plot_maxage_orphanage()


def orphanage_by_time_max_age():
    max_age = [20, 40, 60, 80, 100, 120, 180, 300]
    i = 0
    ors = []
    for age in max_age:
        _, _, _, _, orphanage_df = read_data(data_path_maxage(2, i))
        ors.append(orphanage_df)
        i += 1
    plot_cumulative_orphanage_maxage_by_time(ors, max_age)


def grafana_like_critical_only():
    pass


def closer_look_at_tip_pool_size():
    paths = ["{}/k_{}/orphanage/".format(DATA_PATH, k) for k in Ks]
    mpsi, tips, conf, qs = [], [], [], []
    ks = []
    for i, k in enumerate(Ks):
        mpsi_df, _, tips_df, conf_df, _ = read_data(paths[i])
        mpsi_df, tips_df, conf_df = assign_q_based_on_adv_rate(mpsi_df, tips_df, conf_df)
        mpsi_df, tips_df = merge_nodes_data_with_max(mpsi_df, tips_df)
        crit = K_CRITICAL[k]
        if k == 8:
            crit = 0.9
        filter_q = filter_by_q(tips_df, crit)
        tips_df = tips_df[filter_q]
        if tips_df.empty:
            continue
        tips_df = filter_beginning_tips(tips_df)
        tips.append(tips_df)
        ks.append(k)

    plot_tips_closer_look(tips, ks, K_CRITICAL)


def summary_grafana_like():
    paths = ["{}/k_{}/orphanage/".format(DATA_PATH, k) for k in Ks]
    mpsi, tips, conf, qs = [], [], [], []
    max_tip = 0
    max_conf = 0

    ks = []
    for i, k in enumerate(Ks):
        if k == 16:
            continue
        mpsi_df, _, tips_df, conf_df, _ = read_data(paths[i])
        mpsi_df, tips_df, conf_df = assign_q_based_on_adv_rate(mpsi_df, tips_df, conf_df)
        mpsi_df, tips_df = merge_nodes_data_with_max(mpsi_df, tips_df)

        crit = K_CRITICAL[k]
        if k == 8:
            crit = 0.9
        filter_q = filter_by_q(tips_df, crit)
        tips_df = tips_df[filter_q]
        filter_q = filter_by_q(conf_df, crit)
        conf_df = conf_df[filter_q]

        tips.append(tips_df)
        conf.append(conf_df)
        ks.append(k)

        conf_df_max = exclude_columns(conf_df, ['q', 'Time', ADV_FINALIZATION_COL]).max(axis=0).max()

        # find max value to limit y-axis
        max_tip = max(tips_df['Max Tips'].max(), max_tip)
        max_conf = max(conf_df_max, max_conf)

    plot_tips_final_times_summary('grafana_like_summary', tips, conf, ks, max_tip, max_conf, 12, 'Max Tips', 12)


def summary_infinite():
    ks = [2, 4, 8]
    q_for_k = {
        2: [0.5],
        4: [0.5, 0.75],
        8: [0.75, 0.88],
        16: [0.88, 0.93]
    }
    q_crit_for_k = {
        2: 0.5,
        4: 0.75,
        8: 0.88,
        16: 0.93
    }
    conf = []
    max_conf = 0
    tips = []
    max_tip = 0
    for i, k in enumerate(ks):
        for j, q in enumerate(q_for_k[k]):
            if q_crit_for_k[k] != q:
                continue
            path = data_path_infinite(k, j)
            _, _, tips_df, conf_df, _ = read_data(path)
            conf_df_max = exclude_columns(conf_df, ['Time', ADV_FINALIZATION_COL]).max(axis=0).max()
            max_conf = max(max_conf, conf_df_max)
            conf.append(conf_df)
            tips_df = add_avg_column(tips_df)
            max_tip = max(tips_df["Avg"].max(), max_tip)
            tips.append(tips_df)

    plot_infinite_summary('infinite_summary', tips, conf, ks, 2000, max_conf, 40, 'Avg', 60)


if __name__ == "__main__":
    # the_best_orphanage_start_and_stop_points(orphanage_df)
    # orphanage_by_time()
    # tips_per_q() # useless
    # grafana_like_plots()
    # infinite_tips_plots()
    # infinite_times_plots()
    # max_age_plots()
    # orphanage_by_time_max_age()
    closer_look_at_tip_pool_size()
    # summary_infinite_tips()
    # summary_grafana_like()
    # summary_infinite()
