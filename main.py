from read_data import read_data
from group_data import add_stat_columns, find_the_best_orphanage
from plot_data import plot_tips_by_node, plot_cumulative_orphanage_by_time

DATA_PATH = "data/orphanage/final/k_2"
mpsi_df, mpst_df, tips_df, finalization_df, ORPHANAGE_DF = read_data(DATA_PATH)


def analyse_tips(df):
    plot_tips_by_node(df)


def the_best_orphanage_start_and_stop_points(orphanage_df):
    # K = 2
    qs = [0.5, 0.53, 0.55]
    for q in qs:
        interval, o = find_the_best_orphanage(orphanage_df, q, 50, 5)
        print("Q:,", q, "The best orphanage", o, "found for interval:", interval)


def orphanage_by_time():
    qs = [0.53, 0.55]

    fig = plot_cumulative_orphanage_by_time(ORPHANAGE_DF, qs)


if __name__ == "__main__":
    tips_df = add_stat_columns(tips_df)
    _, _ , _, _, orphanage_df = read_data(DATA_PATH)
    # the_best_orphanage_start_and_stop_points(orphanage_df)

    # analyse_tips(tips_df)
    orphanage_by_time()

