import numpy as np
import pandas as pd
from datetime import timedelta

timeCol = "Time"
advCol = "Tips adversary:9311"
critical_points = {'k=2': [0.5, 0.53, 0.55], 'k=4': []}
MEASUREMENTS_INTERVAL = np.timedelta64(10, 's')

adversary_mpsi_name = "Message Per Second issued by adversary:9311"

# ############## grafana data ######################


def exclude_columns(grafana_df, cols):
    return grafana_df.loc[:, ~grafana_df.columns.isin(cols)]


def add_median_column(df):
    df["Median"] = exclude_columns(df, [timeCol, advCol]).median(axis=1)
    return df


def add_avg_column(df):
    df["Avg"] = exclude_columns(df, [timeCol, advCol]).mean(axis=1)
    return df


def add_max_column(df):
    df["Max"] = exclude_columns(df, [timeCol]).max(axis=1)
    return df


def filter_by_time(df, start, stop):
    f = start < df['Time']
    df = df[f]
    f = df['Time'] <= stop
    return df[f]


def filter_by_adv_issuing_rate(mpsi_df, min_rate):
    return mpsi_df[adversary_mpsi_name] > min_rate


# not really useful, the time ranges dont really fit togheter
def cut_by_q_and_experiment_time_range(orphanage_df, tips_df, mpsi_df, q):
    # get time start and end of experiment from orphanage data
    filtered_by_q = orphanage_df[filter_by_q(orphanage_df, q)]
    experimentStart = filtered_by_q['intervalStart'].min()
    experimentStop = filtered_by_q['intervalStop'].max()
    cut_tips_df = filter_by_time(tips_df, experimentStart, experimentStop)
    # filter by nodes issuing rate is as high as it should be
    return cut_tips_df


# adds column with q value
def assign_q_values(tips_df, mpsi_df, conf_df, orphanage_df):
    tips_df, mpsi_df, conf_df, max_exp_num = cut_by_issue_rate(tips_df, mpsi_df, conf_df)
    qs = get_all_qs(orphanage_df)
    print(qs)
    if len(qs) != max_exp_num:
        raise Exception("There is different number of qs than found experiments in grafana data")
    q_col = tips_df.exp
    for i in range(1, len(qs)+1):
        q_col = q_col.apply(apply_q, args=[i, qs[i-1]])
    tips_df = tips_df.assign(q=q_col)
    mpsi_df = mpsi_df.assign(q=q_col)
    conf_df = conf_df.assign(q=q_col)
    return tips_df, mpsi_df, conf_df


def apply_q(val, i, q):
    if val == i:
        return q
    return val


def cut_by_issue_rate(tips_df, mpsi_df, conf_df):
    # issuing_intervals = filter_by_adv_issuing_rate(mpsi_df, min_rate)
    # tips_df['q'] = 0 > min_rate
    tips_df = tips_df.assign(exp=None)
    conf_df = conf_df.assign(exp=None)
    mpsi_df = mpsi_df.assign(exp=None)

    exp = mpsi_df[adversary_mpsi_name].apply(rolling_count)
    tips_df['exp'] = exp
    conf_df['exp'] = exp
    mpsi_df['exp'] = exp
    max_exp_num = exp.max()
    print("Max range:", max_exp_num)

    return tips_df, mpsi_df, conf_df, max_exp_num


def assign_q_based_on_adv_rate(mpsi, tips, conf):
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
    adv_rate = row[adversary_mpsi_name]
    start_time = row["Time"]
    if pd.isna(adv_rate):
        return prev_qs, start_times
    q = adv_rate / total_mps
    r1 = round(q*2, 1) / 2
    rounded_q = round(r1, 2)
    if len(prev_qs) == 0 or rounded_q > prev_qs[-1]:
        prev_qs.append(rounded_q)
        start_times.append(start_time)
    return prev_qs, start_times


def fill_in_previous_q_rows(df, q, filtered_rows):
    df.loc[filtered_rows, 'q'] = q
    return df


def filter_exp_rows_for_q(df, start_time, duration_in_min):
    end_time = start_time + timedelta(minutes=duration_in_min)
    filter_exp_rows = (start_time <= df['Time']) & (df['Time'] < end_time)
    return filter_exp_rows


def group_tips_by_q(tips_df):
    grouped_df = tips_df.groupby(["q"]).apply(aggregate_tips_by_q)
    grouped_df.reset_index(inplace=True)
    grouped_df = grouped_df.rename(columns={'index': 'q'})
    return grouped_df


def group_times_by_q(conf_df):
    grouped_df = conf_df.groupby(["q"]).apply(aggregate_times_by_q)
    grouped_df.reset_index(inplace=True)
    grouped_df = grouped_df.rename(columns={'index': 'q'})
    return grouped_df


def aggregate_tips_by_q(tips_df):
    result_df = {
        'Tip Pool Size': tips_df["Median"].mean(),
    }
    return pd.Series(result_df, index=['Tip Pool Size'])


def aggregate_times_by_q(conf_df):
    result_df = {
        'Max Finalization Time': conf_df["Max"].max(),
        # 'Confirmed Message Count': conf_df["Max"].count(),
    }
    return pd.Series(result_df, index=['Max Finalization Time'])


def group_conf_times_by_q():
    pass

# ############# orphanage data grouping ####################


def filter_by_requester(df, requester):
    return df["requester"] == requester


def filter_by_range(df, start_interval, stop_interval):
    return start_interval < df["intervalNum"] <= stop_interval


def filter_by_q(df, q):
    return round(df['q'], 2) == round(q, 2)


def filter_by_qs(df, qs):
    return df[df['q'].isin(qs)]


def group_orphanage_by_requester(df):
    df = df.groupby(["intervalNum", "q"]).apply(aggregate_orphanage)
    df.reset_index(inplace=True)
    return df


def group_by_q(df, start_interval, stop_interval):
    measure_filtered = df[df.apply(filter_by_range, args=[start_interval, stop_interval], axis=1)]
    grouped_df = measure_filtered.groupby(["q"]).apply(aggregate_by_q)
    # grouped_df.reset_index(inplace=True)
    # grouped_df = grouped_df.rename(columns={'index': 'q'})
    return grouped_df


def aggregate_orphanage(df):
    orphans = df['honestOrphans'].mean()
    issued = df['honestIssued'].mean()

    result_df = {
        'Orphans': orphans,
        'Issued': issued,
    }

    return pd.Series(result_df, index=['Orphans', 'Issued'])


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


def orphanage_to_time_by_req(orphanage_df, q, requester):
    filter_q = filter_by_q(orphanage_df, q)
    df = orphanage_df[filter_q]
    filter_r = filter_by_requester(df, requester)
    df = orphanage_df[filter_r]


# orphanage plot orphanage rate in time for different qs
def orphanage_to_time(orphanage_df, q, cut_data):
    filter_q = filter_by_q(orphanage_df, q)
    df = orphanage_df[filter_q]
    # need to take experiment start time, before we cut orphanage
    exp_start_time = df['intervalStart'].min()

    # filter by interval, cut the data at the beginning and end of an experiment to get the most possible orphanage rate
    if cut_data:
        (start, stop), _ = find_the_best_orphanage(df, q, 40, 5)
        df = df[df.apply(filter_by_range, args=[start, stop], axis=1)]

    return accumulate_orphans(df, exp_start_time,)


def accumulate_orphans(df, experiment_start):
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

def get_all_qs(orphanage_df):
    q = orphanage_df['q'].apply(round, args=[2]).drop_duplicates()
    return q.values


def get_all_requesters(orphanage_df):
    q = orphanage_df['requester'].drop_duplicates()
    return q.values


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
            grouped_df = group_by_q(filtered_by_q, '', start)
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
    mpsi_df, mpst_df, tips_df, conf_df, orphanage_df = read_data(DATA_PATH)

    df5 = group_by_q(orphanage_df, 5, 40)
    # print("best_range, max_orphanage for 0.5:", find_the_best_orphanage(orphanage_df, 0.5, 50, 5))
    # print("best_range, max_orphanage for 0.53:", find_the_best_orphanage(orphanage_df, 0.53, 50, 5))
    # print("best_range, max_orphanage for 0.55:", find_the_best_orphanage(orphanage_df, 0.55, 50, 5))
    # a = orphanage_df["requester"].drop_duplicates()
    # UWUDXzXPGNk,4pjCdM3LNpC, dAnF7pQ6k7a,GVsK3ww5VAu,4AeXyZ26e4G
    # orphanage_to_time(orphanage_df, 0.55, False)
    # create_orphanage_summary(orphanage_df)

    tips_df, mpsi_df, conf_df = assign_q_values(tips_df, mpsi_df, conf_df, orphanage_df)
    get_all_qs(orphanage_df)
    print("This is the end")
