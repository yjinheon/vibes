
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, TimeSeriesSplit

def stratified_split(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    train_idx, test_idx = train_test_split(df.index, test_size=test_size, random_state=random_state, stratify=df[target])
    return df.loc[train_idx], df.loc[test_idx]

def group_split(df: pd.DataFrame, group_col: str, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df[group_col]))

def timeseries_split(df: pd.DataFrame, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(df))
