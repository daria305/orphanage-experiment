from matplotlib import pyplot as plt

from read_data import read_data
from group_data import add_stat_columns, find_the_best_orphanage
from plot_data import plot_tips_by_node, plot_cumulative_orphanage_by_time

DATA_PATH_K2 = "data/orphanage/final/k_2"
DATA_PATH_K4 = "data/orphanage/final/k_4"
DATA_PATH_K8 = "data/orphanage/final/k_8"


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

    k2_qs = [0.53, 0.55]
    plot_cumulative_orphanage_by_time(orphanage_df_k2, k2_qs)
    k4_qs = [0.65, 0.7, 0.75, 0.8]
    plot_cumulative_orphanage_by_time(orphanage_df_k4, k4_qs)
    k8_qs = [0.65, 0.7, 0.75]
    plot_cumulative_orphanage_by_time(orphanage_df_k8, k8_qs)


if __name__ == "__main__":
    # tips_df = add_stat_columns(tips_df)

    # the_best_orphanage_start_and_stop_points(orphanage_df)

    # analyse_tips(tips_df)
    orphanage_by_time()

