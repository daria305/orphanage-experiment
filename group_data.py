import numpy as np
import pandas as pd

timeCol = "Time"
advCol = "Tips adversary:9311"
critical_points = {'k=2': [0.5, 0.53, 0.55], 'k=4': []}
MAX_PARENT_AGE = np.timedelta64(1, 'm')


# ############## grafana data ######################

def only_honest_df(grafana_df):
    return grafana_df.loc[:, ~grafana_df.columns.isin([timeCol, advCol])]


def add_stat_columns(df):
    df["Median"] = only_honest_df(df).median(axis=1)
    df["Avg"] = only_honest_df(df).median(axis=1)
    return df

# ############# orphanage data grouping ####################


def filter_by_requester(df, requester):
    return df["requester"] == requester


def filter_by_range(df, start_interval, stop_interval):
    if len(df) == 0:
        print("DDDDDDDDDDDD")
    return start_interval < df["intervalNum"] <= stop_interval


def filter_by_q(df, q):
    return df['q'] == q


def group_by_q_per_requester(df, requester, start_interval, stop_interval):
    # req_filtered = df[df.apply(filter_by_requester, args=[requester], axis=1)]
    measure_filtered = df[df.apply(filter_by_range, args=[start_interval, stop_interval], axis=1)]
    grouped_df = measure_filtered.groupby(["q"]).apply(aggregate_by_q)
    # grouped_df.reset_index(inplace=True)
    # grouped_df = grouped_df.rename(columns={'index': 'q'})
    return grouped_df


def aggregate_by_q(df):
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


# orphanage plot orphanage rate in time for different qs
def orphanage_to_time(orphanage_df, q, cut_data):
    filter_q = filter_by_q(orphanage_df, q)
    df = orphanage_df[filter_q]
    # need to take experiment start time, before we cut orphanage
    exp_start_time = df['intervalStart'].min()
    print(exp_start_time)

    # filter by interval, cut the data at the beginning and end of an experiment to get the most possible orphanage rate
    if cut_data:
        (start, stop), _ = find_the_best_orphanage(df, q, 40, 5)
        df = df[df.apply(filter_by_range, args=[start, stop], axis=1)]

    return accumulate_orphans(df, exp_start_time,)


def accumulate_orphans(df, experiment_start):
    orphans = df['honestOrphans'].cumsum()
    issued = df['honestIssued'].cumsum()
    # orphans = df['honestOrphans'].rolling(20).mean()
    # issued = df['honestIssued'].rolling(30).mean()
    cum_rate = orphans/issued

    # can be used if all data was collected at once
    # duration = (df['intervalStop'] - experiment_start) / np.timedelta64(1, 'm')
    duration = df['intervalNum'] * MAX_PARENT_AGE

    result_df = {
        'Orphanage': cum_rate,
        'Time': duration,
    }
    return pd.DataFrame.from_dict(result_df)


# ################# orphanage summary ##################################


# looks for maximum orphanage for a given q, interval start and stop
def find_the_best_orphanage(df, q, start_limit, stop_limit):
    max_interval = df['intervalNum'].max()
    if max_interval < stop_limit:
        stop_limit = max_interval
    max_orphanage = 0
    best_range = (0, 0)
    filtered_by_q = df[filter_by_q(df, q)]
    for start in range(1, start_limit):
        for stop in range(max_interval-stop_limit, max_interval):
            grouped_df = group_by_q_per_requester(filtered_by_q, '', start, stop)
            if len(grouped_df) == 0:
                continue
            max_orph = grouped_df['Orphanage'].max()
            # print(start, stop, max_orph)
            if max_orphanage < max_orph:
                max_orphanage = max_orph
                best_range = (start, stop)
    return best_range, max_orphanage


def create_orphanage_summary(orphanage_df):
    critical_points = {'k=2': [0.5, 0.53, 0.55], 'k=4': []}
    # k=2

    a, c = find_the_best_orphanage(orphanage_df, 0.5, 50, 5)


if __name__ == "__main__":
    DATA_PATH = "data/orphanage/final/k_2"
    from read_data import read_data
    mpsi_df, mpst_df, tips_df, finalization_df, orphanage_df = read_data(DATA_PATH)

    df5 = group_by_q_per_requester(orphanage_df, '', 5, 40)
    # print("best_range, max_orphanage for 0.5:", find_the_best_orphanage(orphanage_df, 0.5, 50, 5))
    # print("best_range, max_orphanage for 0.53:", find_the_best_orphanage(orphanage_df, 0.53, 50, 5))
    # print("best_range, max_orphanage for 0.55:", find_the_best_orphanage(orphanage_df, 0.55, 50, 5))
    # a = orphanage_df["requester"].drop_duplicates()
    # UWUDXzXPGNk,4pjCdM3LNpC, dAnF7pQ6k7a,GVsK3ww5VAu,4AeXyZ26e4G
    orphanage_to_time(orphanage_df, 0.55, False)
    create_orphanage_summary(orphanage_df)
    print("This is the end")
