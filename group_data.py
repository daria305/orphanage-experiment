import numpy as np
import pandas as pd

timeCol = "Time"
advCol = "Tips adversary:9311"
critical_points = {'k=2': [0.5, 0.53, 0.55], 'k=4': []}
MAX_PARENT_AGE = np.timedelta64(1, 'm')

adversary_mpsi_name = "Message Per Second issued by adversary:9311"

# ############## grafana data ######################


def exclude_columns(grafana_df, cols):
    return grafana_df.loc[:, ~grafana_df.columns.isin(cols)]


def add_median_column(df):
    df["Median"] = exclude_columns(df, [timeCol, advCol]).median(axis=1)
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


def rolling_count(adv_tips):
    if adv_tips > rolling_count.min_rate:
        if not rolling_count.previous:
            rolling_count.previous = True
            rolling_count.exp_count += 1
        count = rolling_count.exp_count
    else:
        rolling_count.previous = False
        count = 0
    return count


def setup_rolling_count():
    rolling_count.exp_count = 0  # static variable
    rolling_count.previous = False  # static variable
    rolling_count.min_rate = 2


setup_rolling_count()


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

def get_all_qs(orphanage_df):
    q = orphanage_df['q'].apply(round, args=[2]).drop_duplicates()
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
    mpsi_df, mpst_df, tips_df, conf_df, orphanage_df = read_data(DATA_PATH)

    df5 = group_by_q_per_requester(orphanage_df, '', 5, 40)
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
