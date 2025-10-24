
import pandas as pd

def one_hot(df: pd.DataFrame, cols=None, drop_first=False):
    return pd.get_dummies(df, columns=cols, drop_first=drop_first, dummy_na=False)

def rare_bucket(df: pd.DataFrame, col: str, threshold: float = 0.01, other_label: str = "Other"):
    s = df[col]
    vc = s.value_counts(normalize=True, dropna=False)
    rare = vc[vc < threshold].index
    out = df.copy()
    out[col] = s.where(~s.isin(rare), other_label)
    return out
