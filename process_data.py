import pandas as pd

from group_data import add_avg_column, cut_out_flat_beginning, assign_q_based_on_adv_rate, \
    extend_q_based_on_tip_pool_size, add_max_column, add_median_column


def process_grafana_general(tips: pd.DataFrame, conf: pd.DataFrame, mpsi: pd.DataFrame):
    _, tips, conf = assign_q_based_on_adv_rate(mpsi, tips, conf)
    tips, conf = extend_q_based_on_tip_pool_size(tips, conf)

    tips = add_avg_column(tips)
    tips = add_max_column(tips)
    tips = add_median_column(tips)

    tips, conf = cut_out_flat_beginning(tips, conf)
    return tips, conf


def process_tips_infinite(df: pd.DataFrame):
    pass


def process_tips_closer(df: pd.DataFrame):
    pass