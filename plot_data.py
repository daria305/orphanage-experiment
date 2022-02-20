import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from group_data import orphanage_to_time, exclude_columns, TIME_COL, EXP_DURATION, filter_by_q, \
    ADV_FINALIZATION_COL, ADV_TIPS_COL, create_duration_axis

# Graphs properties

LINE_WIDTH = 2
LINE_WIDTH_THIN = 1
COLORS = sns.color_palette(n_colors=8)
# COLORS = sns.hls_palette(10, h=0.5)
LINE_TYPE = ['-', '-.', '- -', ':']

FIG_SIZE = (16, 8)
FIG_SIZE_SHORT = (16, 6)
FIG_SIZE_HIGH = (12, 8)
MARKER_SIZE = 3

SMALL_SIZE = 14
MEDIUM_SIZE = 16

orphanage_filename = "orphanage_by_time"
SAVE_FORMAT = 'pdf'
SAVE_DIR = 'plots/'


K_CRITICAL = {
    2: 0.5,
    4: 0.75,
    8: 0.88,
    16: 0.94
}
MEASUREMENTS_INTERVAL = np.timedelta64(10, 's')
MAX_AGE_MEASUREMENT_INTERVALS = [np.timedelta64(d, 'ms') for d in [1666, 3333, 5000, 6666, 8333, 10000, 15000, 25000]]


def save_results_to_csv(df, filename):
    df.to_csv(filename, index=True, header=False)


def plot_tips_by_node(df):
    df.plot("Time", "Avg", linewidth=LINE_WIDTH, color=COLORS[0])
    # plt.show()


# ####################### orphanage ####################################

def plot_cumulative_orphanage_by_time(df: pd.DataFrame, qs, file_name):
    plt.figure(figsize=FIG_SIZE_SHORT)
    for i, q in enumerate(qs):
        df_per_q = orphanage_to_time(df, q, MEASUREMENTS_INTERVAL)
        x = df_per_q['Time'] / np.timedelta64(1, 'm')
        y = df_per_q['Orphanage']
        plt.plot(x, y, label="q={}".format(q),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".", markersize=MARKER_SIZE)

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig(SAVE_DIR + file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    # plt.show()


def plot_orphanage_by_time_summary(filename, dfs: [pd.DataFrame], subplot_details, ks):
    plt.figure(figsize=FIG_SIZE)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=FIG_SIZE_SHORT, constrained_layout=True)
    # critical
    for subplot_num in range(len(subplot_details)):
        for i, k in enumerate(ks):
            for j, q in enumerate(subplot_details[subplot_num][k]):
                line_style = 'solid'
                if q == K_CRITICAL[k]:
                    line_style = 'dashed'
                df_per_q = orphanage_to_time(dfs[i], q, MEASUREMENTS_INTERVAL)
                x = df_per_q['Time'] / np.timedelta64(1, 'm')
                y = df_per_q['Orphanage']
                axes[subplot_num].plot(x, y, linewidth=LINE_WIDTH, color=COLORS[i], label="k={}, q={}".format(k, q),
                             linestyle=line_style)
        axes[subplot_num].legend()
        axes[subplot_num].set_xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
        if subplot_num == 0:
            axes[subplot_num].set_ylabel("Orphanage", fontsize=MEDIUM_SIZE)
        plt.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_maxage_summary(tips: [pd.DataFrame], confs: [pd.DataFrame], orphanages: [pd.DataFrame], ages, q):
    filename = "maxage_summary"
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=FIG_SIZE_HIGH, constrained_layout=True)
    skip_exp = [1, 3, 4]

    # orphanage
    labels = []
    color_id = 0
    for i, df in enumerate(ages):
        df = orphanage_to_time(orphanages[i], q, MAX_AGE_MEASUREMENT_INTERVALS[i])
        x = df['Time'] / np.timedelta64(1, 'm')
        y = df['Orphanage']
        axes[0].plot(x, y, linewidth=LINE_WIDTH, color=COLORS[color_id])
        labels.append(r"$\zeta$={}s".format(ages[i]))
        color_id += 1

    axes[0].legend(labels)
    # tips
    color_id = 0
    for i, df in enumerate(tips):
        if i in skip_exp:
            continue
        experiment_duration = ages[i] * 10 / 60
        y = df["Median"]
        x = pd.Series(np.linspace(0, experiment_duration, num=len(y)))
        label = r'$\zeta$={}'.format(ages[i])
        axes[1].plot(x, y, linewidth=LINE_WIDTH, label=label, color=COLORS[color_id])
        color_id += 1

    # conf
    color_id = 0
    for i, df in enumerate(confs):
        # [20, 40, 60, 80, 100, 120, 180, 300]
        if i in skip_exp:
            continue
        experiment_duration = ages[i] * 10 / 60
        y = df["Moving Avg"]
        x = pd.Series(np.linspace(0, experiment_duration, num=len(y)))
        label = r'$\zeta$={}'.format(ages[i])
        axes[2].plot(x, y / float(1000000000 * 60), linewidth=LINE_WIDTH, label=label, color=COLORS[color_id], )
        color_id += 1

    axes[0].set_ylabel("Orphanage", fontsize=SMALL_SIZE)
    axes[1].set_ylabel("Tip Pool Size", fontsize=SMALL_SIZE)
    axes[2].set_ylabel("Finalization Time", fontsize=SMALL_SIZE)
    axes[0].set_xlim([0, 35])
    axes[1].set_xlim([0, 35])
    axes[2].set_xlim([0, 35])

    axes[2].set_xlabel(r"Attack duration [min]: $10\cdot\zeta$", fontsize=SMALL_SIZE)

    plt.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_cumulative_orphanage_maxage_by_time(dfs: [pd.DataFrame], ages, q):
    filename = "maxage_orphanage"
    plt.figure(figsize=FIG_SIZE_SHORT)
    color_id = 0
    for i, df in enumerate(dfs):
        df = orphanage_to_time(dfs[i], q, MAX_AGE_MEASUREMENT_INTERVALS[i])
        x = df['Time'] / np.timedelta64(1, 'm')
        y = df['Orphanage']
        plt.plot(x, y, label=r"$\zeta$={}s".format(ages[i]),
                 linewidth=LINE_WIDTH, color=COLORS[color_id])
        color_id += 1

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel(r"Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    # plt.show()


def plot_grafana_tips_q_for_all_k(ks, tips_dfs):
    filename = "median_tips_per_q"
    limit_q_top = 1.01
    limit_bottom = 0.1
    plt.figure(figsize=FIG_SIZE)
    for i, df in enumerate(tips_dfs):
        filtered_df = df[df.q < limit_q_top]
        filtered_df = filtered_df[limit_bottom < filtered_df.q]
        a = filtered_df['q']
        b = filtered_df['Tip Pool Size']
        plt.plot(a, b, label="k={}".format(ks[i]),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".", markersize=MARKER_SIZE)
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("q", fontsize=MEDIUM_SIZE)
    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    plt.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)

    # plt.show()


def plot_grafana_times_q_for_all_k(ks, times_dfs):
    filename = "max_conf_time_per_q"
    limit_q_top = 1.01
    limit_bottom = 0.1
    plt.figure(figsize=FIG_SIZE)
    for i, df in enumerate(times_dfs):
        filtered_df = df[df.q < limit_q_top]
        filtered_df = filtered_df[limit_bottom < filtered_df.q]
        a = filtered_df['q']
        b = filtered_df['Median Finalization Time']
        plt.plot(a, b, label="k={}".format(ks[i]),
                 linewidth=0, color=COLORS[i], marker=".", markersize=MARKER_SIZE)
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("q", fontsize=MEDIUM_SIZE)
    plt.ylabel("Finalization Time [min]", fontsize=MEDIUM_SIZE)
    plt.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)

    # plt.show()


# ################## grafana like plots ############################


def plot_tips_final_times(tips_df: pd.DataFrame, conf_df: pd.DataFrame, k, tip_y_limit, conf_y_limit):
    grafana_plot_k_q = {
        2: [0.35, 0.4, 0.45, 0.5, 0.55],
        4: [0.6, 0.65, 0.7, 0.75, 0.8],
        8: [0.7, 0.75, 0.8, 0.88, 0.93],
        16: [0.8, 0.85, 0.9, 0.94, 0.99],
    }
    plot_grafana_tips_subplot_per_q(tips_df, k, grafana_plot_k_q[k], tip_y_limit)
    plot_grafana_conf_subplot_per_q(conf_df, k, grafana_plot_k_q[k], conf_y_limit)


def plot_grafana_tips_subplot_per_q(df: pd.DataFrame, k, qs, y_limit):
    filename = "grafana_like_tips_k"
    # qs = qs[k]
    # each q has its own subplot
    fig, axes = plt.subplots(nrows=1, ncols=len(qs), figsize=FIG_SIZE_SHORT, constrained_layout=True)
    for subplot_num in range(len(qs)):
        q = qs[subplot_num]
        filtered_df = filter_by_q(df, k, q)
        if df.empty:
            continue
        filtered_df = create_duration_axis(filtered_df, 'minute')
        x = filtered_df['duration'].reset_index(drop=True)
        y = filtered_df["Max"]
        axes[subplot_num].plot(x, y, linewidth=LINE_WIDTH, color=COLORS[subplot_num])
        axes[subplot_num].set_xlabel("q={}".format(qs[subplot_num]), fontsize=SMALL_SIZE)
        axes[subplot_num].set_ylim([0, y_limit])

    for i, ax in enumerate(axes):
        if i != 0:
            # hide y-axes
            ax.yaxis.set_visible(False)
        else:
            ax.set_ylabel("Tip Pool Size", fontsize=SMALL_SIZE)

    plt.savefig(SAVE_DIR + filename + str(k) + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    # plt.show()


def plot_grafana_conf_subplot_per_q(df: pd.DataFrame, k, qs, y_limit):
    filename = "grafana_like_conf_time_k"
    # each q has its own subplot
    y_limit = y_limit / float(1000000000 * 60)
    fig, axes = plt.subplots(nrows=1, ncols=len(qs), figsize=FIG_SIZE_SHORT, constrained_layout=True)
    conf_cols = exclude_columns(df, [ADV_FINALIZATION_COL, TIME_COL, 'q']).columns

    for subplot_num in range(len(qs)):
        q = qs[subplot_num]
        for i, col in enumerate(conf_cols):
            df_filtered = filter_by_q(df, k, q)
            df_filtered = create_duration_axis(df_filtered, 'minute')
            x = df_filtered['duration'].reset_index(drop=True)
            y = df_filtered[col] / float(1000000000 * 60)
            axes[subplot_num].plot(x, y, linewidth=0, color=COLORS[subplot_num], marker=".", markersize=MARKER_SIZE)
            axes[subplot_num].set_ylim([0, y_limit])

        axes[subplot_num].set_xlabel("q={}".format(qs[subplot_num]), fontsize=SMALL_SIZE)

    for i, ax in enumerate(axes):
        if i != 0:
            # hide y-axes
            ax.yaxis.set_visible(False)
        else:
            ax.set_ylabel("Confirmation times [min]", fontsize=SMALL_SIZE)

    plt.savefig(SAVE_DIR + filename + str(k) + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    # plt.show()


def plot_grafana_q_variations_summary(tips: pd.DataFrame, confs: pd.DataFrame, qs: [float], k: int, tips_limit: int,
                                      conf_limit: int):
    filename = "grafana_like_q_vary_summary"
    fig, axes = plt.subplots(nrows=2, ncols=len(qs), figsize=FIG_SIZE, constrained_layout=True)

    for subplot_num in range(len(qs)):
        q = qs[subplot_num]
        filtered_df = filter_by_q(tips, k, q)
        filtered_df = create_duration_axis(filtered_df, 'minute')
        x = filtered_df['duration'].reset_index(drop=True)
        y = filtered_df["Max"]
        axes[0][subplot_num].plot(x, y, linewidth=LINE_WIDTH, color=COLORS[subplot_num])
        axes[0][subplot_num].set_ylim([0, tips_limit])

    conf_cols = exclude_columns(confs, [ADV_FINALIZATION_COL, TIME_COL, 'q']).columns
    conf_limit = conf_limit / float(1000000000 * 60)

    for subplot_num, q in enumerate(qs):
        for i, col in enumerate(conf_cols):
            df_filtered = filter_by_q(confs, k, q)
            df_filtered = create_duration_axis(df_filtered, 'minute')
            x = df_filtered['duration'].reset_index(drop=True)
            y = df_filtered[col] / float(1000000000 * 60)
            axes[1][subplot_num].plot(x, y, linewidth=0, color=COLORS[subplot_num], marker=".", markersize=MARKER_SIZE)
            axes[1][subplot_num].set_ylim([0, conf_limit])
        axes[1][subplot_num].set_xlabel("k=2, q={}".format(qs[subplot_num]), fontsize=SMALL_SIZE)

    axes[0][0].set_ylabel("Tip Pool Size", fontsize=SMALL_SIZE)
    axes[1][0].set_ylabel("Confirmation times [min]", fontsize=SMALL_SIZE)
    plt.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_tips_closer_look(tips_dfs: [pd.DataFrame], ks, q_per_k):
    file_name = "tips_closer_look"
    plt.figure(figsize=FIG_SIZE)
    points_60 = []
    points_120 = []
    for i, df in enumerate(tips_dfs):
        df = create_duration_axis(df, 'second')
        x = df['duration'].reset_index(drop=True)
        y = df["Max"].reset_index(drop=True)
        plt.plot(x, y, linewidth=LINE_WIDTH, label="k={}, q={}".format(ks[i], q_per_k[ks[i]]),
                 color=COLORS[i])

        idx = df.loc[df['duration'] == 60].index.values
        points_60.append((60, df.loc[idx, 'Max']))
        idx = df.loc[df['duration'] == 120].index.values
        points_120.append((120, df.loc[idx, 'Max']))

    for i, point in enumerate(points_60):
        plt.scatter(point[0], point[1], marker='.', c=COLORS[i])
        plt.annotate(' (%.0f, %.0f)' % (point[0], point[1]), xy=(point[0] - 8, point[1] + 3))

    for i, point in enumerate(points_120):
        plt.scatter(point[0], point[1], marker='.', c=COLORS[i])
        plt.annotate(' (%.0f, %.0f)' % (point[0], point[1]), xy=(point[0], point[1] - 40))

    total_rate = 50

    for i, k in enumerate(ks):
        x60 = 60
        x120 = 120
        p60 = theoretical_tip_pool_size(x60, k, q_per_k[k], total_rate)
        p120 = theoretical_tip_pool_size(x120, k, q_per_k[k], total_rate)
        plt.scatter(x60, p60, marker='.', c=COLORS[i])
        plt.scatter(x120, p120, marker='.', c=COLORS[i])
        plt.plot([0, x120], [0, p120], linewidth=LINE_WIDTH_THIN, color=COLORS[i], linestyle=':')

    plt.plot([-1, -1], [-1, -1], linewidth=LINE_WIDTH_THIN, color='k', linestyle=':', marker='.', label='theoretical values')
    plt.xlim([0, 130])
    plt.ylim([0, 1800])
    plt.legend(loc='upper left', fontsize=MEDIUM_SIZE)
    plt.xlabel("Time [s]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    # plt.show()
    plt.savefig(SAVE_DIR + file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def theoretical_tip_pool_size(t, k, q: float, total_rate):
    return t * total_rate *(q - (1-q)*(k-1)) + 1


def plot_tips_final_times_summary(filename, tips_dfs: [pd.DataFrame], conf_dfs: [pd.DataFrame], ks, tip_y_limit,
                                  conf_y_limit, x_limit, tips_col_name, exp_duration):
    grafana_plot_correct_q_values = {
        2: 0.5,
        4: 0.75,
        8: 0.88,
        16: 0.94
    }
    fig, axes = plot_grafana_tips_subplot_per_q_summary(filename, tips_dfs, conf_dfs, ks, grafana_plot_correct_q_values,
                                                        tip_y_limit, conf_y_limit, x_limit, tips_col_name, exp_duration)
    fig.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_infinite_summary(filename, tips_dfs: [pd.DataFrame], conf_dfs: [pd.DataFrame], ks, tip_y_limit,
                          conf_y_limit, x_limit, tips_col_name, exp_duration):
    grafana_plot_correct_q_values = {
        2: 0.5,
        4: 0.75,
        8: 0.88,
        16: 0.94
    }
    fig, axes = plot_grafana_tips_subplot_per_q_summary(filename, tips_dfs, conf_dfs, ks, grafana_plot_correct_q_values,
                                                        tip_y_limit, conf_y_limit, x_limit, tips_col_name, exp_duration)
    fig, axes = add_lines_infinite_summary(fig, axes)
    fig.savefig(SAVE_DIR + filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_grafana_tips_subplot_per_q_summary(filename, tips_dfs: pd.DataFrame, conf_dfs: pd.DataFrame, ks, q_corrections,
                                            tips_limit, conf_limit, x_limit, tips_col_name, exp_duration):
    # each q has its own subplot
    qs_labels = [q_corrections[k] for k in ks]
    fig, axes = plt.subplots(nrows=2, ncols=len(ks), figsize=FIG_SIZE, constrained_layout=True)
    conf_cols = exclude_columns(conf_dfs[0], [ADV_FINALIZATION_COL, TIME_COL, 'q']).columns

    for subplot_num, df in enumerate(tips_dfs):
        df = create_duration_axis(df, 'minute')
        x = df['duration'].reset_index(drop=True)
        y = df[tips_col_name]
        axes[0][subplot_num].plot(x, y, linewidth=LINE_WIDTH, color=COLORS[subplot_num])
        axes[0][subplot_num].set_xlabel("k={}, q={}".format(ks[subplot_num], qs_labels[subplot_num]),
                                        fontsize=SMALL_SIZE)
        axes[0][subplot_num].set_ylim([0, tips_limit])
        axes[0][subplot_num].set_xlim([0, x_limit])

    conf_limit = conf_limit / float(1000000000 * 60)
    for subplot_num, df in enumerate(conf_dfs):
        df = create_duration_axis(df, 'minute')
        x = df['duration'].reset_index(drop=True)
        for i, col in enumerate(conf_cols):
            labels = ["k={}, q={}".format(ks[subplot_num], qs_labels[subplot_num])]
            labels.extend([""] * (len(conf_cols) - 1))
            y = df[col] / float(1000000000 * 60)
            axes[1][subplot_num].plot(x, y, linewidth=0, color=COLORS[subplot_num], marker=".", markersize=MARKER_SIZE)
            axes[1][subplot_num].set_ylim([0, conf_limit])
            axes[1][subplot_num].set_xlim([0, x_limit])

        axes[1][subplot_num].set_xlabel("k={}, q={}".format(ks[subplot_num], qs_labels[subplot_num]),
                                        fontsize=SMALL_SIZE)

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            pass
            if j != 0:
                # hide y-axes
                ax.yaxis.set_visible(False)
            else:
                if i == 0:
                    ax.set_ylabel("Tip Pool Size", fontsize=SMALL_SIZE)
                if i == 1:
                    ax.set_ylabel("Finalization Times [min]", fontsize=SMALL_SIZE)
            if i != 1:
                ax.xaxis.set_visible(False)
    # plt.show()
    return fig, axes


def add_lines_infinite_summary(fig, axes):
    points = ((37.5, 1223), (20.6, 1223), (5.5, 1380), (3.2, 1050))
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            xy = points[j]
            ax.hlines(xy[1], -20, xy[0], linestyles='--', colors='grey', linewidth=LINE_WIDTH_THIN)
            ax.vlines(xy[0], -20, xy[1], linestyles='--', colors='grey', linewidth=LINE_WIDTH_THIN)
            ax.scatter(xy[0], xy[1], marker='.')
            if i == 0:
                if j < 2:
                    ax.annotate('(%.0f, %.0f)' % xy, xy=[xy[0] - 8, xy[1] + 13])
                else:
                    ax.annotate('(%.0f, %.0f)' % xy, xy=[xy[0], xy[1] + 15])
    return fig, axes


# ################### Infinite parent age check ###########################


def plot_tips_infinite(tips_dfs, k, qs, y_limit):
    file_name = "infinite-tips-critical_k_{}".format(k)
    grafana_time_diff = float(1) / 12  # minutes
    plt.figure(figsize=FIG_SIZE_SHORT)

    for i, df in enumerate(tips_dfs):
        df = df.assign(duration=grafana_time_diff)
        df['duration'] = df['duration'].cumsum().apply(lambda x: x)
        q = qs[i]
        plt.plot(df['duration'], df['Avg'], linewidth=LINE_WIDTH, label="q={}".format(q), color=COLORS[i])
        plt.ylim([0, 2000])

        plt.legend(loc='best', fontsize=MEDIUM_SIZE)
        plt.xlabel("Time [min]", fontsize=MEDIUM_SIZE)
        plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
        plt.xlim([0, 60])
        plt.savefig(SAVE_DIR + file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_times_infinite(times_dfs, k, qs, y_limit):
    grafana_time_diff = float(1) / 12  # minutes
    plt.figure(figsize=FIG_SIZE_SHORT)
    file_name = "infinite-times-critical_k_{}".format(k)
    y_limit = y_limit / float(1000000000 * 60)

    for i, df in enumerate(times_dfs):
        df = df.assign(duration=grafana_time_diff)
        df['duration'] = df['duration'].cumsum().apply(lambda x: x)
        conf_cols = exclude_columns(df, [TIME_COL, ADV_FINALIZATION_COL, 'q', 'duration']).columns
        q = qs[i]
        for j, col in enumerate(conf_cols):
            labels = ["q={}".format(q)]
            labels.extend([""] * (len(conf_cols) - 1))
            plt.plot(df['duration'], df[col] / float(1000000000 * 60), linewidth=0, label=labels[j],
                     color=COLORS[i], marker='.', markersize=MARKER_SIZE)
            plt.ylim([0, y_limit])
        plt.legend(loc='best', fontsize=MEDIUM_SIZE)

        plt.ylabel("Confirmation times [min]", fontsize=MEDIUM_SIZE)
        plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
        plt.xlim([0, 60])
        plt.savefig(SAVE_DIR + file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_maxage_tips(max_age, tips):
    plt.figure(figsize=FIG_SIZE)
    file_name = "maxage_tips"
    c = 0

    for i, df in enumerate(tips):
        age = max_age[i]
        experiment_duration = age * 10 / 60
        y = df["Median"]
        x = pd.Series(np.linspace(0, experiment_duration, num=len(y)))
        label = r'$\zeta$={}'.format(age)
        plt.plot(x, y, linewidth=LINE_WIDTH, label=label, color=COLORS[c])
        # plt.ylim([0, 2000])
        c += 1
    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.savefig(SAVE_DIR + file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_maxage_conf(max_age, conf):
    plt.figure(figsize=FIG_SIZE)
    file_name = "maxage_conf"
    c = 0
    for i, df in enumerate(conf):
        # [20, 40, 60, 80, 100, 120, 180, 300]
        if i in [1, 3, 4]:
            continue
        age = max_age[i]
        experiment_duration = age * 10 / 60
        y = df["Moving Avg"]
        x = pd.Series(np.linspace(0, experiment_duration, num=len(y)))
        label = r'$\zeta$={}'.format(age)
        plt.plot(x, y / float(1000000000 * 60), linewidth=LINE_WIDTH, label=label, color=COLORS[c], )
        # plt.ylim([0, 2000])
        c += 1
    plt.ylabel("Confirmation times [min]", fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.savefig(SAVE_DIR + file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_maxage_conf_separate_nodes(max_age, tips):
    plt.figure(figsize=FIG_SIZE)
    file_name = "maxage_conf"

    for i, df in enumerate(tips):
        if i in [1, 3, 4, 6]:
            continue
        age = max_age[i]
        experiment_duration = age * 10
        conf_cols = exclude_columns(df, [TIME_COL, ADV_FINALIZATION_COL]).columns
        for j, col in enumerate(conf_cols):
            labels = [r'$\zeta$={}'.format(age)]
            labels.extend([""] * (len(conf_cols) - 1))
            y = df[col]
            x = pd.Series(np.linspace(0, EXP_DURATION, num=len(y)))
            plt.plot(x, df[col] / float(1000000000 * 60), linewidth=0, label=labels[j],
                     color=COLORS[i], marker='.', markersize=MARKER_SIZE)
        # plt.ylim([0, 2000])

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.savefig(SAVE_DIR + file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
