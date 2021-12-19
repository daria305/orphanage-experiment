import matplotlib.pyplot as plt
import seaborn as sns

from group_data import orphanage_to_time, exclude_columns, timeCol, advCol

# Graphs properties

LINE_WIDTH = 2
COLORS = sns.color_palette()
LINE_TYPE = ['-', '-.', '- -', ':']
FIG_SIZE = (16, 8)
MARKER_SIZE = 10

SMALL_SIZE = 11
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

orphanage_filename = "orphanage_by_time"

grafana_time_diff = float(1)/6  # minutes


def save_results_to_csv(df, filename):
    df.to_csv(filename, index=True, header=False)


def plot_tips_by_node(df):
    df.plot("Time", "Avg", linewidth=LINE_WIDTH, color=COLORS[0])
    plt.show()


# ####################### orphanage ####################################

def plot_cumulative_orphanage_by_time(df, qs, file_name):
    plt.figure(figsize=FIG_SIZE)
    for i, q in enumerate(qs):
        df_per_q = orphanage_to_time(df, q, False)
        a = df_per_q['Time']
        b = df_per_q['Orphanage']
        print(type(a), type(b))
        plt.plot(df_per_q['Time'], df_per_q['Orphanage'], label="q={}".format(q),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")

    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig(file_name + '.pdf', format='pdf')
    # plt.show()


def plot_grafana_tips_q_for_all_k(ks, tips_dfs):
    limit_q_top = 0.8
    plt.figure(figsize=FIG_SIZE)
    for i, df in enumerate(tips_dfs):
        filtered_df = df[df.q < limit_q_top]
        a = filtered_df['q']
        b = filtered_df['Tip Pool Size']
        plt.plot(a, b, label="k={}".format(ks[i]),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("q", fontsize=MEDIUM_SIZE)
    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    plt.savefig("median_tips_per_q" + '.pdf', format='pdf')

    # plt.show()


def plot_grafana_times_q_for_all_k(ks, times_dfs):
    limit_q_bottom = 0.3
    limit_q_top = 0.8
    plt.figure(figsize=FIG_SIZE)
    for i, df in enumerate(times_dfs):
        filtered_df = df[df.q > limit_q_bottom]
        filtered_df = filtered_df[filtered_df.q < limit_q_top]
        a = filtered_df['q']
        b = filtered_df['Max Finalization Time']
        plt.plot(a, b, label="k={}".format(ks[i]),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")
    plt.legend(loc='best', fontsize=MEDIUM_SIZE)
    plt.xlabel("q", fontsize=MEDIUM_SIZE)
    plt.ylabel("Max Finalization Time [min]", fontsize=MEDIUM_SIZE)
    plt.savefig("max_conf_time_per_q" + '.pdf', format='pdf')

    # plt.show()


# ################## grafana like plots ############################


def plot_issuance(mpsi_df):
    pass


def plot_tips_final_times(tips_df, conf_df, k):

    tips_df = tips_df.assign(duration=grafana_time_diff)
    tips_df['duration'] = tips_df['duration'].cumsum().apply(lambda x: int(x))

    conf_df = conf_df.assign(duration=grafana_time_diff)
    conf_df['duration'] = conf_df['duration'].cumsum().apply(lambda x: int(x))

    tips_cols = exclude_columns(tips_df, [timeCol, advCol, 'exp', 'q', 'duration']).columns
    conf_cols = exclude_columns(conf_df, [timeCol, "Msg finalization ", 'exp', 'q', 'duration']).columns

    plt.figure(figsize=FIG_SIZE)
    for i, col in enumerate(tips_cols):
        plt.plot(tips_df['duration'], tips_df[col], linewidth=1, color=COLORS[i])

    plt.ylabel("Tip Pool Size", fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.savefig("grafana_like_tips_k" + str(k) + '.pdf', format='pdf')
    # plt.show()

    plt.figure(figsize=FIG_SIZE)
    for i, col in enumerate(conf_cols):
        plt.plot(conf_df['duration'], conf_df[col], color=COLORS[i], linewidth=0, marker='.')

    plt.ylabel("Confirmation times [min]", fontsize=MEDIUM_SIZE)
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)

    plt.savefig("grafana_like_conf_time_k" + str(k) + '.pdf', format='pdf')

    # plt.show()
