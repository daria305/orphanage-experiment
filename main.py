import pandas as pd

from process_data import process_grafana_general
from read_data import read_data
from group_data import add_median_column, add_max_column, \
    group_tips_by_q, group_times_by_q, add_avg_column, \
    assign_q_based_on_adv_rate, get_all_qs, exclude_columns, ADV_FINALIZATION_COL, \
    add_moving_avg_column, idle_spam_time_end, cut_by_time, filter_by_q, filter_beginning_tips, \
    extend_q_based_on_tip_pool_size, cut_out_flat_beginning, get_limit
from plot_data import plot_tips_by_node, plot_cumulative_orphanage_by_time, plot_grafana_tips_q_for_all_k, \
    plot_grafana_times_q_for_all_k, plot_tips_final_times, plot_tips_infinite, plot_times_infinite, plot_maxage_tips, \
    plot_maxage_conf, plot_cumulative_orphanage_maxage_by_time, plot_tips_closer_look, plot_tips_final_times_summary, \
    plot_infinite_summary, plot_orphanage_by_time_summary, plot_maxage_summary, plot_grafana_q_variations_summary

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

K_CRITICAL = {
    2: 0.5,
    4: 0.75,
    8: 0.88,
    16: 0.93
}

# parameters selection for different plots

ORPHANAGE_K_Q = {
    2: [0.5, 0.53, 0.55],
    4: [0.7, 0.75, 0.8],
    8: [0.8, 0.88, 0.93],
    16: [0.9, 0.94, 0.99]
}

INFINITE_K_Q = {
    2: [0.5],
    4: [0.5, 0.75],
    8: [0.75, 0.88],
    16: [0.88, 0.93]
}

CLOSER_LOOK_K_Q = {
    2: 0.55,
    4: 0.80,
    8: 0.93,
}


def orphanage_by_time():
    for k in Ks:
        _, _, _, _, orphanage_df = read_data(data_path_orphanage(k, ""))
        plot_cumulative_orphanage_by_time(orphanage_df, ORPHANAGE_K_Q[k], "orphanage_k{}".format(k))


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
        tips_df, conf_df = process_grafana_general(tips_df, conf_df, mpsi_df)

        tips.append(tips_df)
        conf.append(conf_df)

        # find max value to limit y-axis
        conf_df_max = get_limit(conf_df)
        max_tip = max(tips_df['Max'].max(), max_tip)
        max_conf = max(conf_df_max, max_conf)

    for i, k in enumerate(Ks):
        plot_tips_final_times(tips[i], conf[i], k, max_tip, max_conf)


def infinite_times_plots():
    ks = [2, 4, 8, 16]

    conf = []
    max_conf = 0
    for i, k in enumerate(ks):
        conf_dfs = []
        for j, q in enumerate(INFINITE_K_Q[k]):
            path = data_path_infinite(k, j)
            mpsi_df, _, tips_df, conf_df, _ = read_data(path)
            tips_df, conf_df = process_grafana_general(tips_df, conf_df, mpsi_df)

            conf_df_max = get_limit(conf_df)
            max_conf = max(max_conf, conf_df_max)
            conf_dfs.append(conf_df)

        conf.append(conf_dfs)
    for i, k in enumerate(ks):
        plot_times_infinite(conf[i], k, INFINITE_K_Q[k], max_conf)


def infinite_tips_plots():
    ks = [2, 4, 8, 16]
    tips = []
    max_tip = 0
    for i, k in enumerate(ks):
        tips_dfs = []
        for j, q in enumerate(INFINITE_K_Q[k]):
            path = data_path_infinite(k, j)
            mpsi_df, _, tips_df, conf_df, _ = read_data(path)
            tips_df, _ = process_grafana_general(tips_df, conf_df, mpsi_df)

            max_tip = max(tips_df["Avg"].max(), max_tip)
            tips_dfs.append(tips_df)

        tips.append(tips_dfs)
    for i, k in enumerate(ks):
        plot_tips_infinite(tips[i], k, INFINITE_K_Q[k], max_tip)


def max_age_plots():
    max_age = [20, 40, 60, 80, 100, 120, 180, 300]
    q = 0.5

    tips = []
    confs = []
    orphanages = []
    for i, age in enumerate(max_age):
        path = data_path_maxage(2, i)
        mpsi_df, _, tips_df, conf_df, orphanage_df = read_data(path)
        tips_df, _ = process_grafana_general(tips_df, conf_df, mpsi_df)

        conf_df = add_moving_avg_column(conf_df, 10)
        tips.append(tips_df)
        confs.append(conf_df)
        orphanages.append(orphanage_df)

    plot_maxage_tips(max_age, tips)
    plot_maxage_conf(max_age, confs)
    plot_cumulative_orphanage_maxage_by_time(orphanages, max_age, q)
    plot_maxage_summary(tips, confs, orphanages, max_age, q)


def grafana_like_critical_only():
    pass


def closer_look_at_tip_pool_size():
    ks = [2, 4, 8]
    paths = ["{}/k_{}/orphanage/".format(DATA_PATH, k) for k in ks]

    mpsi, tips, conf, qs = [], [], [], []
    for i, k in enumerate(ks):
        mpsi_df, _, tips_df, conf_df, _ = read_data(paths[i])
        tips_df, conf_df = process_grafana_general(tips_df, conf_df, mpsi_df)
        tips_df = filter_by_q(tips_df, k, CLOSER_LOOK_K_Q[k])
        tips_df = filter_beginning_tips(tips_df)
        tips.append(tips_df)

    plot_tips_closer_look(tips, ks, CLOSER_LOOK_K_Q)


def summary_grafana_like():
    ks = [2, 4, 8]
    paths = ["{}/k_{}/orphanage/".format(DATA_PATH, k) for k in Ks]
    mpsi, tips, conf, qs = [], [], [], []
    max_tip = 0
    max_conf = 0

    for i, k in enumerate(ks):
        mpsi_df, _, tips_df, conf_df, _ = read_data(paths[i])
        tips_df, conf_df = process_grafana_general(tips_df, conf_df, mpsi_df)

        tips_df = filter_by_q(tips_df, k, K_CRITICAL[k])
        conf_df = filter_by_q(conf_df, k, K_CRITICAL[k])

        tips.append(tips_df)
        conf.append(conf_df)

        conf_df_max = get_limit(conf_df)

        # find max value to limit y-axis
        max_tip = max(tips_df['Max'].max(), max_tip)
        max_conf = max(conf_df_max, max_conf)

    plot_tips_final_times_summary('grafana_like_summary', tips, conf, ks, max_tip, max_conf, 12, 'Max', 12)


def summary_infinite():
    ks = [2, 4, 8]

    conf = []
    max_conf = 0
    tips = []
    max_tip = 0

    for i, k in enumerate(ks):
        for j, q in enumerate(INFINITE_K_Q[k]):
            if K_CRITICAL[k] != q:
                continue
            path = data_path_infinite(k, j)
            mpsi_df, _, tips_df, conf_df, _ = read_data(path)
            tips_df, conf_df = process_grafana_general(tips_df, conf_df, mpsi_df)

            # calculate limit values
            conf_df_max = get_limit(conf_df)
            max_conf = max(max_conf, conf_df_max)
            max_tip = max(tips_df["Avg"].max(), max_tip)

            tips.append(tips_df)
            conf.append(conf_df)

    plot_infinite_summary('infinite_summary', tips, conf, ks, 2000, max_conf, 40, 'Avg', 60)


def orphanage_summary():
    sub1 = {
        2: [0.5],
        4: [0.5, 0.75],
        8: [0.75, 0.88],
        16: [0.9, 0.94]
    }
    sub2 = {
        2: [0.55],
        4: [0.8],
        8: [0.93],
        16: [0.99]
    }
    ks = [2, 4, 8, 16]
    orphanages = []
    for k in ks:
        _, _, _, _, orphanage_df = read_data(data_path_orphanage(k, ""))
        orphanages.append(orphanage_df)

    plot_orphanage_by_time_summary('orphanage_summary', orphanages, (sub1, sub2), ks)


def grafana_like_q_variations():
    path = "{}/k_2/orphanage/".format(DATA_PATH)
    k = 2
    qs = [0.35, 0.45, 0.5, 0.55]
    max_tip = 0
    max_conf = 0

    mpsi_df, _, tips_df, conf_df, _ = read_data(path)
    tips_df, conf_df = process_grafana_general(tips_df, conf_df, mpsi_df)

    # find max value to limit y-axis
    conf_df_max = get_limit(conf_df)
    max_tip = max(tips_df['Max'].max(), max_tip)
    max_conf = max(conf_df_max, max_conf)

    plot_grafana_q_variations_summary(tips_df, conf_df, qs, k, max_tip, max_conf)


if __name__ == "__main__":
    # appendix
    grafana_like_plots()
    orphanage_by_time()
    infinite_tips_plots()
    infinite_times_plots()
    # base
    max_age_plots()
    summary_grafana_like()
    summary_infinite()
    closer_look_at_tip_pool_size()
    orphanage_summary()
    grafana_like_q_variations()
