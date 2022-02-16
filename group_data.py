import numpy as np
import pandas as pd
from datetime import timedelta

TIME_COL = "Time"
ADV_TIPS_COL = "Tips adversary:9311"
ADV_FINALIZATION_COL = "Msg finalization adversary:9311"
ADV_MPSI_COL = "Message Per Second issued by adversary:9311"

MEASUREMENTS_INTERVAL = np.timedelta64(10, 's')
EXP_DURATION = 12


# ############## grafana data ######################


def exclude_columns(grafana_df: pd.DataFrame, cols: [str]):
    return grafana_df.loc[:, ~grafana_df.columns.isin(cols)]


def add_median_column(df: pd.DataFrame):
    df["Median"] = exclude_columns(df, [TIME_COL, ADV_TIPS_COL]).median(axis=1)
    return df


def add_avg_column(df: pd.DataFrame):
    df["Avg"] = exclude_columns(df, [TIME_COL, ADV_TIPS_COL]).mean(axis=1)
    return df


def add_moving_avg_column(df: pd.DataFrame, window_size):
    df = exclude_columns(df, [TIME_COL, ADV_FINALIZATION_COL])
    df["Median"] = df.median(axis=1)
    df["Moving Avg"] = df["Median"].rolling(window_size).median(axis=1)
    return df


def add_max_column(df: pd.DataFrame):
    df["Max"] = exclude_columns(df, [TIME_COL]).max(axis=1)
    return df


def keep_only_columns(grafana_df: pd.DataFrame, cols: [str]):
    return grafana_df.loc[:, grafana_df.columns.isin(cols)]


def merge_nodes_data_with_max(mpsi, tips):
    """
    :type mpsi: pd.DataFrame
    :type tips: pd.DataFrame
    """

    mpsi["Max Rate"] = exclude_columns(mpsi, [TIME_COL, ADV_MPSI_COL, 'q']).max(axis=1)
    tips["Max Tips"] = exclude_columns(tips, [TIME_COL, ADV_TIPS_COL, 'q']).max(axis=1)

    mpsi = keep_only_columns(mpsi, ["Max Rate", "q", "Time"])
    tips = keep_only_columns(tips, ["Max Tips", "q", "Time"])

    return mpsi, tips


def assign_q_based_on_adv_rate(mpsi, tips, conf):
    """
    :type mpsi: pd.DataFrame
    :type tips: pd.DataFrame
    :type conf: pd.DataFrame
    """
    duration = 12
    total_mps = 50
    # tips, mpsi, conf,
    qs = []
    start_times = []
    # get starting row index for each q based on adversary rate and q proportion

    for _, row in mpsi.iterrows():
        qs, start_times = calculate_q(row, total_mps, qs, start_times)
    # insert q columns
    mpsi = mpsi.assign(q=0.)
    tips = tips.assign(q=0.)
    conf = conf.assign(q=0.)
    for start_time, q in zip(start_times, qs):
        filtered_rows = filter_exp_rows_for_q(mpsi, start_time, duration)
        mpsi = fill_in_previous_q_rows(mpsi, q, filtered_rows)

        filtered_rows = filter_exp_rows_for_q(tips, start_time, duration)
        tips = fill_in_previous_q_rows(tips, q, filtered_rows)

        filtered_rows = filter_exp_rows_for_q(conf, start_time, duration)
        conf = fill_in_previous_q_rows(conf, q, filtered_rows)\

    return mpsi, tips, conf


def calculate_q(row, total_mps, prev_qs, start_times):
    adv_rate = row[ADV_MPSI_COL]
    start_time = row["Time"]
    if pd.isna(adv_rate):
        return prev_qs, start_times
    q = adv_rate / total_mps
    # round q to 0.05 accuracy
    r1 = round(q*2, 1) / 2
    rounded_q = round(r1, 2)
    if len(prev_qs) == 0 or rounded_q > prev_qs[-1]:
        prev_qs.append(rounded_q)
        start_times.append(start_time)
    return prev_qs, start_times


def fill_in_previous_q_rows(df: pd.DataFrame, q, filtered_rows):
    df.loc[filtered_rows, 'q'] = q
    return df


def filter_exp_rows_for_q(df: pd.DataFrame, start_time, duration_in_min):
    end_time = start_time + timedelta(minutes=duration_in_min)
    filter_exp_rows = (start_time <= df['Time']) & (df['Time'] < end_time)
    return filter_exp_rows


def group_tips_by_q(tips: pd.DataFrame):
    grouped_df = tips.groupby(["q"]).apply(aggregate_tips_by_q)
    grouped_df.reset_index(inplace=True)
    grouped_df = grouped_df.rename(columns={'index': 'q'})
    return grouped_df


def group_times_by_q(conf: pd.DataFrame):
    grouped_df = conf.groupby(["q"]).apply(aggregate_times_by_q)
    grouped_df.reset_index(inplace=True)
    grouped_df = grouped_df.rename(columns={'index': 'q'})
    return grouped_df


def aggregate_tips_by_q(tips: pd.DataFrame):
    result_df = {
        'Tip Pool Size': tips["Median"].max(),
    }
    return pd.Series(result_df, index=['Tip Pool Size'])


def aggregate_times_by_q(conf: pd.DataFrame):
    result_df = {
        'Median Finalization Time': conf["Median"].max(),
        # 'Confirmed Message Count': conf_df["Max"].count(),
    }
    return pd.Series(result_df, index=['Median Finalization Time'])


# ############# orphanage data grouping ####################


def filter_by_requester(df, requester):
    return df["requester"] == requester


def filter_by_range(df: pd.DataFrame, start_interval, stop_interval):
    return start_interval < df["intervalNum"] <= stop_interval


def filter_by_q(df: pd.DataFrame, q):
    return round(df['q'], 2) == round(q, 2)


def filter_by_qs(df: pd.DataFrame, qs):
    return df[df['q'].isin(qs)]


def group_orphanage_by_requester(df: pd.DataFrame):
    df = df.groupby(["intervalNum", "q"]).apply(aggregate_orphanage)
    df.reset_index(inplace=True)
    return df


def group_by_q(df: pd.DataFrame, start_interval, stop_interval):
    measure_filtered = df[df.apply(filter_by_range, args=(start_interval, stop_interval), axis=1)]
    grouped_df = measure_filtered.groupby(["q"]).apply(aggregate_by_q)
    # grouped_df.reset_index(inplace=True)
    # grouped_df = grouped_df.rename(columns={'index': 'q'})
    return grouped_df


def aggregate_orphanage(df: pd.DataFrame):
    orphans = df['honestOrphans'].mean()
    issued = df['honestIssued'].mean()

    result_df = {
        'Orphans': orphans,
        'Issued': issued,
    }

    return pd.Series(result_df, index=['Orphans', 'Issued'])


def aggregate_by_q(df: pd.DataFrame):
    orphans = df['honestOrphans'].sum()
    issued = df['honestIssued'].sum()

    result_df = {
        'Orphans': orphans,
        'Issued': issued,
        'Orphanage': orphans/issued,
        'Interval Start': df['intervalNum'].min(),
        'Interval Stop': df['intervalNum'].max(),
        'Duration': df['intervalStop'].max() - df['intervalStart'].min()
    }

    return pd.Series(result_df, index=['Orphans', 'Issued', 'Orphanage', 'Interval Start', 'Interval Stop', 'Duration'])


def filter_beginning_tips(tips_df_with_single_q: pd.DataFrame):
    startTime = tips_df_with_single_q[TIME_COL].iloc[0]
    cutTime = startTime + timedelta(minutes=1)
    filter_df = tips_df_with_single_q[TIME_COL] < cutTime
    return tips_df_with_single_q[filter_df]


# orphanage plot orphanage rate in time for different qs
def orphanage_to_time(orphanage: pd.DataFrame, q, cut_data):
    filter_q = filter_by_q(orphanage, q)
    df = orphanage[filter_q]

    # filter by interval, cut the data at the beginning and end of an experiment to get the most possible orphanage rate
    if cut_data:
        (start, stop), _ = find_the_best_orphanage(df, q, 40, 5)
        df = df[df.apply(filter_by_range, args=[start, stop], axis=1)]

    return accumulate_orphans(df)


def accumulate_orphans(df: pd.DataFrame):
    df = group_orphanage_by_requester(df)
    orphans = df['Orphans'].cumsum()
    issued = df['Issued'].cumsum()
    # orphans = grouped_req['honestOrphans'].rolling(20).mean()
    # issued = grouped_req['honestIssued'].rolling(30).mean()
    cum_rate = orphans/issued

    # can be used if all data was collected at once
    # duration = (df['intervalStop'] - experiment_start) / np.timedelta64(1, 'm')
    duration = df['intervalNum'] * MEASUREMENTS_INTERVAL

    result_df = {
        'Orphanage': cum_rate,
        'Time': duration,
    }
    return pd.DataFrame.from_dict(result_df)


# ################# orphanage summary ##################################

def get_all_qs(df: pd.DataFrame):
    q = df['q'].apply(round, args=[2]).drop_duplicates()
    return q.values


def get_all_requesters(orphanage: pd.DataFrame):
    q = orphanage['requester'].drop_duplicates()
    return q.values


# looks for maximum orphanage for a given q, interval start and stop
def find_the_best_orphanage(df: pd.DataFrame, q, start_limit, stop_limit):
    max_interval = df['intervalNum'].max()
    if max_interval < stop_limit:
        stop_limit = max_interval
    max_orphanage = 0
    best_range = (0, 0)
    filtered_by_q = df[filter_by_q(df, q)]
    for start in range(1, start_limit):
        for stop in range(max_interval-stop_limit, max_interval):
            grouped_df = group_by_q(filtered_by_q, '', start)
            if len(grouped_df) == 0:
                continue
            max_orphan = grouped_df['Orphanage'].max()
            if max_orphanage < max_orphan:
                max_orphanage = max_orphan
                best_range = (start, stop)
    return best_range, max_orphanage


def idle_spam_time_end(df_mpsi):
    if df_mpsi[ADV_MPSI_COL].loc[0] > 1:
        return df_mpsi['Time'].loc[0]
    ddf = df_mpsi[1][df_mpsi[df_mpsi[1] > 1]]
    return ddf.loc[0]


def cut_by_time(df, cut_time):
    df = df[df['Time'] > cut_time]
    return df


if __name__ == "__main__":
    DATA_PATH = "data/orphanage/final/k_2"
    from read_data import read_data
    mpsi_df, mpst_df, tips_df, conf_df, orphanage_df = read_data(DATA_PATH)

    df5 = group_by_q(orphanage_df, 5, 40)
    # print("best_range, max_orphanage for 0.5:", find_the_best_orphanage(orphanage_df, 0.5, 50, 5))
    # print("best_range, max_orphanage for 0.53:", find_the_best_orphanage(orphanage_df, 0.53, 50, 5))
    # print("best_range, max_orphanage for 0.55:", find_the_best_orphanage(orphanage_df, 0.55, 50, 5))
    # a = orphanage_df["requester"].drop_duplicates()
    # orphanage_to_time(orphanage_df, 0.55, False)
    # create_orphanage_summary(orphanage_df)

    get_all_qs(orphanage_df)
    print("This is the end")
