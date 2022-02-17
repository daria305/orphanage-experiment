import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from group_data import orphanage_to_time, exclude_columns, TIME_COL, EXP_DURATION, filter_by_q, \
    ADV_FINALIZATION_COL, ADV_TIPS_COL, create_duration_axis

# Graphs properties


LINE_WIDTH = 2
COLORS = sns.color_palette(n_colors=20)
LINE_TYPE = ['-', '-.', '- -', ':']
FIG_SIZE = (16, 8)
FIG_SIZE_SHORT = (16, 6)
MARKER_SIZE = 10

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

orphanage_filename = "orphanage_by_time"
SAVE_FORMAT = 'pdf'


def save_results_to_csv(df, filename):
    df.to_csv(filename, index=True, header=False)


def plot_tips_by_node(df):
    df.plot("Time", "Avg", linewidth=LINE_WIDTH, color=COLORS[0])
    # plt.show()


# ####################### orphanage ####################################

def plot_cumulative_orphanage_by_time(df: pd.DataFrame, qs, file_name):
    plt.figure(figsize=FIG_SIZE_SHORT)
    for i, q in enumerate(qs):
        df_per_q = orphanage_to_time(df, q)
        x = df_per_q['Time'] / np.timedelta64(1, 'm')
        y = df_per_q['Orphanage']
        plt.plot(x, y, label="q={}".format(q),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig(file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    # plt.show()


def plot_orphanage_by_time_summary(filename, dfs: [pd.DataFrame], subplot_details, ks):
    plt.figure(figsize=FIG_SIZE_SHORT)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=FIG_SIZE_SHORT, constrained_layout=True)
    # critical
    for i, k in enumerate(ks):
        for j, q in enumerate(subplot_details[0][k]):
            df_per_q = orphanage_to_time(dfs[i], q)
            x = df_per_q['Time'] / np.timedelta64(1, 'm')
            y = df_per_q['Orphanage']
            plt.plot(x, y, label="k={}, q={}".format(k, q),
                     linewidth=LINE_WIDTH, color=COLORS[i], marker=".", linetype=LINE_TYPE[j])

    # above critical
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig(filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_cumulative_orphanage_maxage_by_time(dfs: [pd.DataFrame], ages):
    plt.figure(figsize=FIG_SIZE_SHORT)
    for i, df in enumerate(dfs):
        if i in [1, 3, 4, 6]:
            continue
        df = create_duration_axis(df, 'minute')
        x = df['duration']
        y = df['honestOrphanageRate']
        plt.plot(x, y, label="max age={}".format(ages[i]),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig("maxage_orphanage" + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    plt.show()


def plot_grafana_tips_q_for_all_k(ks, tips_dfs):
    limit_q_top = 1.01
    limit_bottom = 0.1
    plt.figure(figsize=FIG_SIZE)
    for i, df in enumerate(tips_dfs):
        filtered_df = df[df.q < limit_q_top]
        filtered_df = filtered_df[limit_bottom < filtered_df.q]
        a = filtered_df['q']
        b = filtered_df['Tip Pool Size']
        plt.plot(a, b, label="k={}".format(ks[i]),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("q", fontsize=MEDIUM_SIZE)
    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    plt.savefig("median_tips_per_q" + '.' + SAVE_FORMAT, format=SAVE_FORMAT)

    # plt.show()


def plot_grafana_times_q_for_all_k(ks, times_dfs):
    limit_q_top = 1.01
    limit_bottom = 0.1
    plt.figure(figsize=FIG_SIZE)
    for i, df in enumerate(times_dfs):
        filtered_df = df[df.q < limit_q_top]
        filtered_df = filtered_df[limit_bottom < filtered_df.q]
        a = filtered_df['q']
        b = filtered_df['Median Finalization Time']
        plt.plot(a, b, label="k={}".format(ks[i]),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("q", fontsize=MEDIUM_SIZE)
    plt.ylabel("Finalization Time [min]", fontsize=MEDIUM_SIZE)
    plt.savefig("max_conf_time_per_q" + '.' + SAVE_FORMAT, format=SAVE_FORMAT)

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
    # qs = qs[k]
    # each q has its own subplot
    fig, axes = plt.subplots(nrows=1, ncols=len(qs), figsize=FIG_SIZE_SHORT, constrained_layout=True)
    for subplot_num in range(len(qs)):
        q = qs[subplot_num]
        df = filter_by_q(df, k, q)
        if df.empty:
            continue
        df = create_duration_axis(df, 'minute')
        x = df['duration'].reset_index(drop=True)
        y = df["Max"]
        axes[subplot_num].plot(x, y, linewidth=1, color=COLORS[subplot_num])
        axes[subplot_num].set_xlabel("q={}".format(qs[subplot_num]), fontsize=SMALL_SIZE)
        axes[subplot_num].set_ylim([0, y_limit])

    for i, ax in enumerate(axes):
        if i != 0:
            # hide y-axes
            ax.yaxis.set_visible(False)
        else:
            ax.set_ylabel("Tip Pool Size", fontsize=SMALL_SIZE)

    plt.savefig("grafana_like_tips_k" + str(k) + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    # plt.show()


def plot_grafana_conf_subplot_per_q(df: pd.DataFrame, k, qs, y_limit):
    # each q has its own subplot
    y_limit = y_limit / float(1000000000 * 60)
    qs_labels = [q for q in qs]
    fig, axes = plt.subplots(nrows=1, ncols=len(qs), figsize=FIG_SIZE_SHORT, constrained_layout=True)
    conf_cols = exclude_columns(df, [ADV_FINALIZATION_COL, TIME_COL, 'q']).columns

    for subplot_num in range(len(qs)):
        q = qs[subplot_num]
        for i, col in enumerate(conf_cols):
            df_filtered = filter_by_q(df, k, q)
            df_filtered = create_duration_axis(df_filtered, 'minute')
            x = df_filtered['duration'].reset_index(drop=True)
            y = df_filtered[col] / float(1000000000 * 60)
            axes[subplot_num].plot(x, y, linewidth=0, color=COLORS[subplot_num], marker=".")
            axes[subplot_num].set_ylim([0, y_limit])

        axes[subplot_num].set_xlabel("q={}".format(qs_labels[subplot_num]), fontsize=SMALL_SIZE)

    for i, ax in enumerate(axes):
        if i != 0:
            # hide y-axes
            ax.yaxis.set_visible(False)
        else:
            ax.set_ylabel("Confirmation times [min]", fontsize=SMALL_SIZE)

    plt.savefig("grafana_like_conf_time_k" + str(k) + '.' + SAVE_FORMAT, format=SAVE_FORMAT)
    # plt.show()


def plot_tips_closer_look(tips_dfs: [pd.DataFrame], ks, q_per_k):
    file_name = "tips_closer_look"
    plt.figure(figsize=FIG_SIZE)

    for i, df in enumerate(tips_dfs):
        df = create_duration_axis(df, 'second')
        x = df['duration'].reset_index(drop=True)
        y = df["Max"].reset_index(drop=True)

        plt.plot(x, y, linewidth=1, label="k={}, q={}".format(ks[i], q_per_k[ks[i]]),
                 color=COLORS[i])
        m = y.max()
        idx_max = x[y.idxmax()]
        plt.hlines(m, -20, idx_max, linestyles='--', colors=COLORS[i], linewidth=1)
        plt.scatter(idx_max, m, marker='.')
        plt.annotate(' (%.0f, %.0f)' % (idx_max, m), xy=(idx_max, m+4))

    points_60 = [(60, 261), (60, 674), (60, 1099)]
    for i, point in enumerate(points_60):
        plt.scatter(point[0], point[1], marker='.', c=COLORS[i])
        plt.annotate(' (%.0f, %.0f)' % (point[0], point[1]), xy=(point[0]-8, point[1]+3))
    plt.xlim([0, 130])
    plt.legend(loc='upper left', fontsize=MEDIUM_SIZE)
    plt.xlabel("Time [s]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    # plt.show()
    plt.savefig(file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


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
    fig.savefig(filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_infinite_summary(filename, tips_dfs: [pd.DataFrame], conf_dfs: [pd.DataFrame], ks, tip_y_limit,
                                  conf_y_limit, x_limit, tips_col_name, exp_duration):
    grafana_plot_correct_q_values = {
        2: 0.5,
        4: 0.75,
        8: 0.88,
    }
    fig, axes = plot_grafana_tips_subplot_per_q_summary(filename, tips_dfs, conf_dfs, ks, grafana_plot_correct_q_values,
                                                        tip_y_limit, conf_y_limit, x_limit, tips_col_name, exp_duration)
    fig, axes = add_lines_infinite_summary(fig, axes)
    fig.savefig(filename + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


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
        axes[0][subplot_num].plot(x, y, linewidth=1, color=COLORS[subplot_num])
        axes[0][subplot_num].set_xlabel("k={}, q={}".format(ks[subplot_num], qs_labels[subplot_num]), fontsize=SMALL_SIZE)
        axes[0][subplot_num].set_ylim([0, tips_limit])
        axes[0][subplot_num].set_xlim([0, x_limit])

    conf_limit = conf_limit / float(1000000000 * 60)
    for subplot_num, df in enumerate(conf_dfs):
        df = create_duration_axis(df, 'minute')
        x = df['duration'].reset_index(drop=True)
        for i, col in enumerate(conf_cols):
            labels = ["k={}, q={}".format(ks[subplot_num], qs_labels[subplot_num])]
            labels.extend([""]*(len(conf_cols)-1))
            y = df[col] / float(1000000000 * 60)
            axes[1][subplot_num].plot(x, y, linewidth=0, color=COLORS[subplot_num], marker=".")
            axes[1][subplot_num].set_ylim([0, conf_limit])
            axes[1][subplot_num].set_xlim([0, x_limit])

        axes[1][subplot_num].set_xlabel("k={}, q={}".format(ks[subplot_num], qs_labels[subplot_num]), fontsize=SMALL_SIZE)

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
    points = ((37.5, 1223), (20.6, 1223), (5.5, 1380))
    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            xy = points[j]
            ax.hlines(xy[1], -20, xy[0], linestyles='--', colors='grey', linewidth=1)
            ax.vlines(xy[0], -20, xy[1], linestyles='--', colors='grey', linewidth=1)
            ax.scatter(xy[0], xy[1], marker='.')
            if i == 0:
                if j < 2:
                    ax.annotate('(%.0f, %.0f)' % xy, xy=[xy[0]-6, xy[1]+13])
                else:
                    ax.annotate('(%.0f, %.0f)' % xy, xy=[xy[0], xy[1]+13])
    return fig, axes


# ################### Infinite parent age check ###########################


# todo get median of nodes data
def plot_tips_infinite(tips_dfs, k, qs, y_limit):
    file_name = "infinite-tips-critical_k_{}".format(k)
    grafana_time_diff = float(1)/12  # minutes
    plt.figure(figsize=FIG_SIZE_SHORT)

    for i, df in enumerate(tips_dfs):
        df = df.assign(duration=grafana_time_diff)
        df['duration'] = df['duration'].cumsum().apply(lambda x: x)
        tips_cols = exclude_columns(df, [TIME_COL, ADV_TIPS_COL, 'q', 'duration']).columns
        q = qs[i]
        plt.plot(df['duration'], df['Avg'], linewidth=1, label="q={}".format(q), color=COLORS[i])
        plt.ylim([0, 2000])

        plt.legend(loc='best', fontsize=MEDIUM_SIZE)
        plt.xlabel("Time [min]", fontsize=MEDIUM_SIZE)
        plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
        plt.xlim([0, 60])
        plt.savefig(file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_times_infinite(times_dfs, k, qs, y_limit):
    grafana_time_diff = float(1)/12  # minutes
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
            labels.extend([""]*(len(conf_cols)-1))
            plt.plot(df['duration'], df[col] / float(1000000000 * 60), linewidth=0, label=labels[j],
                     color=COLORS[i], marker='.', )
            plt.ylim([0, y_limit])
        plt.legend(loc='best', fontsize=MEDIUM_SIZE)

        plt.ylabel("Confirmation times [min]", fontsize=MEDIUM_SIZE)
        plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
        plt.xlim([0, 60])
        plt.savefig(file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_maxage_tips(max_age, tips):
    plt.figure(figsize=FIG_SIZE)
    file_name = "maxage_tips"
    c = 0

    for i, df in enumerate(tips):
        age = max_age[i]
        experiment_duration = age * 10
        if i in [1, 3, 4, 6]:
            continue
        y = df["Median"]
        x = pd.Series(np.linspace(0, EXP_DURATION, num=len(y)))
        label = "MaxParentAge={}".format(age)
        plt.plot(x, y, linewidth=1, label=label, color=COLORS[c])
        # plt.ylim([0, 2000])
        c += 1

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.savefig(file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


def plot_maxage_conf(max_age, conf):
    plt.figure(figsize=FIG_SIZE)
    file_name = "maxage_conf"
    c= 0
    for i, df in enumerate(conf):
        # [20, 40, 60, 80, 100, 120, 180, 300]
        if i in [1, 3, 4, 6]:
            continue
        age = max_age[i]
        experiment_duration = age * 10
        y = df["Moving Avg"]
        x = pd.Series(np.linspace(0, EXP_DURATION, num=len(y)))
        label = "MaxParentAge={}".format(age)
        plt.plot(x, y / float(1000000000 * 60), linewidth=1, label=label, color=COLORS[c], )
        # plt.ylim([0, 2000])
        c += 1

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.savefig(file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)


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
            labels = ["MaxParentAge={}".format(age)]
            labels.extend([""]*(len(conf_cols)-1))
            y = df[col]
            x = pd.Series(np.linspace(0, EXP_DURATION, num=len(y)))
            plt.plot(x, df[col] / float(1000000000 * 60), linewidth=0, label=labels[j],
                     color=COLORS[i], marker='.', )
        # plt.ylim([0, 2000])

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.savefig(file_name + '.' + SAVE_FORMAT, format=SAVE_FORMAT)

