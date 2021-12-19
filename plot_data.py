import matplotlib.pyplot as plt
import seaborn as sns

from group_data import orphanage_to_time

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


def save_results_to_csv(df, filename):
    df.to_csv(filename, index=True, header=False)


def plot_tips_by_node(df):
    df.plot("Time", "Avg", linewidth=LINE_WIDTH, color=COLORS[0])
    plt.show()


# ####################### orphanage ####################################

def plot_cumulative_orphanage_by_time(df, qs):
    plt.figure(figsize=FIG_SIZE)
    for i, q in enumerate(qs):
        df_per_q = orphanage_to_time(df, q, False)
        print()
        plt.plot(df_per_q['Time'], df_per_q['Orphanage'], label="q={}".format(q),
                 linewidth=LINE_WIDTH, color=COLORS[i], marker=".")

    plt.legend(loc='best')
    plt.xlabel("Attack duration [min]", fontsize=MEDIUM_SIZE)
    plt.ylabel("Orphanage", fontsize=MEDIUM_SIZE)
    plt.savefig(orphanage_filename + '.pdf', format='pdf')
    plt.show()
