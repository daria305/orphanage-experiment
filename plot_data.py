import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from group_data import orphanage_to_time, exclude_columns, TIME_COL, EXP_DURATION, filter_by_q, \
    ADV_FINALIZATION_COL

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


def save_results_to_csv(df, filename):
    df.to_csv(filename, index=True, header=False)


def plot_tips_by_node(df):
    df.plot("Time", "Avg", linewidth=LINE_WIDTH, color=COLORS[0])
    # plt.show()


# ####################### orphanage ####################################

def plot_cumulative_orphanage_by_time(df: pd.DataFrame, qs, file_name):
    plt.figure(figsize=FIG_SIZE_SHORT)
    for i, q in enumerate(qs):
        df_per_q = orphanage_to_time(df, q, False)
        a = df_per_q['Time'] / np.timedelta64(1, 'm')
        b = df_per_q['Orphanage']
        plt.plot(a, b, label="q={}".format(q),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig(file_name + '.pdf', format='pdf')
    # plt.show()


def plot_grafana_tips_q_for_all_k(ks, tips_dfs):
    limit_q_top = 1.01
    limit_bottom = 0.1
    plt.figure(figsize=FIG_SIZE)
    for i, df in enumerate(tips_dfs):
        filtered_df = df[df.q < limit_q_top]
        filtered_df = filtered_df[limit_bottom < filtered_df.q]
        a = filtered_df['q']
        b = filtered_df['Tip Pool Size']
        print(a.drop_duplicates())
        plt.plot(a, b, label="k={}".format(ks[i]),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("q", fontsize=MEDIUM_SIZE)
    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    plt.savefig("median_tips_per_q" + '.pdf', format='pdf')

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
    plt.savefig("max_conf_time_per_q" + '.pdf', format='pdf')

    # plt.show()


# ################## grafana like plots ############################

def plot_tips_final_times(tips_df: pd.DataFrame, conf_df: pd.DataFrame, k, tip_y_limit, conf_y_limit):
    grafana_plot_k_q = {
        2: [0.35, 0.4, 0.45, 0.5, 0.55],
        4: [0.6, 0.65, 0.7, 0.75, 0.8],
        8: [0.7, 0.75, 0.8, 0.9, 0.95],
        16: [0.8, 0.85, 0.9, 0.95, 1.0],
    }
    grafana_plot_correct_q_values = {
        2: {},
        4: {},
        8: {
            0.9: 0.88,
            0.95: 0.93,
        },
        16: {
            0.95: 0.93,
            1: 0.99,
        }
    }
    plot_grafana_tips_subplot_per_q(tips_df, k, grafana_plot_k_q[k], grafana_plot_correct_q_values[k], tip_y_limit)
    plot_grafana_conf_subplot_per_q(conf_df, k, grafana_plot_k_q[k], grafana_plot_correct_q_values[k], conf_y_limit)


def plot_grafana_tips_subplot_per_q(df: pd.DataFrame, k, qs, q_corrections, y_limit):
    # each q has its own subplot
    qs_to_use = qs
    qs_labels = [q_corrections[q] if q in q_corrections else q for q in qs_to_use]
    fig, axes = plt.subplots(nrows=1, ncols=len(qs_to_use), figsize=FIG_SIZE_SHORT, constrained_layout=True)
    for subplot_num in range(len(qs_to_use)):

        q = qs_to_use[subplot_num]

        y = df[filter_by_q(df, q)]["Max Tips"]
        x = pd.Series(np.linspace(0, EXP_DURATION, num=len(y)))
        axes[subplot_num].plot(x, y, linewidth=1, color=COLORS[subplot_num])
        axes[subplot_num].set_xlabel("q={}".format(qs_labels[subplot_num]), fontsize=SMALL_SIZE)
        axes[subplot_num].set_ylim([0, y_limit])

    for i, ax in enumerate(axes):
        if i != 0:
            # hide y-axes
            ax.yaxis.set_visible(False)
        else:
            ax.set_ylabel("Tip Pool Size", fontsize=SMALL_SIZE)

    plt.savefig("grafana_like_tips_k" + str(k) + '.pdf', format='pdf')
    # plt.show()


def plot_grafana_conf_subplot_per_q(df: pd.DataFrame, k, qs, q_corrections, y_limit):
    # each q has its own subplot
    y_limit = y_limit / float(1000000000 * 60)
    qs_to_use = qs
    qs_labels = [q_corrections[q] if q in q_corrections else q for q in qs_to_use]
    fig, axes = plt.subplots(nrows=1, ncols=len(qs_to_use), figsize=FIG_SIZE_SHORT, constrained_layout=True)
    conf_cols = exclude_columns(df, [ADV_FINALIZATION_COL, TIME_COL, 'q']).columns

    for subplot_num in range(len(qs_to_use)):
        q = qs_to_use[subplot_num]
        for i, col in enumerate(conf_cols):
            y = df[filter_by_q(df, q)][col] / float(1000000000 * 60)
            x = pd.Series(np.linspace(0, EXP_DURATION, num=len(y)))
            axes[subplot_num].plot(x, y, linewidth=0, color=COLORS[subplot_num], marker=".")
            axes[subplot_num].set_ylim([0, y_limit])

        axes[subplot_num].set_xlabel("q={}".format(qs_labels[subplot_num]), fontsize=SMALL_SIZE)

    for i, ax in enumerate(axes):
        if i != 0:
            # hide y-axes
            ax.yaxis.set_visible(False)
        else:
            ax.set_ylabel("Confirmation times [min]", fontsize=SMALL_SIZE)

    plt.savefig("grafana_like_conf_time_k" + str(k) + '.pdf', format='pdf')
    # plt.show()

# ################### Infinite parent age check ###########################


def plot_tips_infinite(tips_dfs: [pd.DataFrame], ks, qs):
    file_name = "infinite-tips-critical"
    grafana_time_diff = float(1)/6  # minutes
    plt.figure(figsize=FIG_SIZE)

    for i, df in enumerate(tips_dfs):
        df = df.assign(duration=grafana_time_diff)
        df['duration'] = df['duration'].cumsum().apply(lambda x: x)

        a = df['duration']
        b = df['Avg']
        plt.plot(a, b, label="k={}, q={}".format(ks[i], qs[i]), linewidth=LINE_WIDTH, color=COLORS[i], marker=".")

        plt.legend(loc='best', fontsize=MEDIUM_SIZE)
        plt.xlabel("Time [min]", fontsize=MEDIUM_SIZE)
        plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
        plt.xlim([0, 125])
        plt.savefig(file_name + '.pdf', format='pdf')


def plot_times_infinite(times_dfs, k, qs):
    grafana_time_diff = float(1)/6  # minutes
    plt.figure(figsize=FIG_SIZE)
    file_name = "infinite-times-critical_k_{}".format(k)

    for i, df in enumerate(times_dfs):
        df = df.assign(duration=grafana_time_diff)
        df['duration'] = df['duration'].cumsum().apply(lambda x: x)
        conf_cols = exclude_columns(df, [TIME_COL, "Msg finalization ", 'exp', 'q', 'duration']).columns
        q = qs[i]
        for j, col in enumerate(conf_cols):
            labels = ["q={}".format(q)]
            labels.extend([""]*(len(conf_cols)-1))
            plt.plot(df['duration'], df[col] / float(1000000000 * 60), linewidth=0, label=labels[j],
                     color=COLORS[i], marker='.', )
        plt.legend(loc='best', fontsize=MEDIUM_SIZE)

        plt.ylabel("Confirmation times [min]", fontsize=MEDIUM_SIZE)
        plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)

        plt.savefig(file_name + '.pdf', format='pdf')
