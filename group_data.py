import pandas as pd

timeCol = "Time"
advCol = "Tips adversary:9311"


def only_honest_df(df):
    return df.loc[:, ~df.columns.isin([timeCol, advCol])]


def add_stat_columns(df):
    df["Median"] = only_honest_df(df).median(axis=1)
    df["Avg"] = only_honest_df(df).median(axis=1)
    return df


if __name__ == "__main__":
    DATA_PATH = "data/orphanage/equalSnapshot/30mpsDur8"
    from read_data import read_data
    mpsi_df, mpst_df, tips_df, finalization_df, orphanage_df = read_data(DATA_PATH)
    tips_df = add_stat_columns(tips_df)
    mpsi_df = add_stat_columns(mpsi_df)
    mpst_df = add_stat_columns(mpst_df)
    print(tips_df.header())