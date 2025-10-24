
import pandas as pd
from pandas.api.types import is_numeric_dtype

def correlation_numeric(df: pd.DataFrame, method: str = "pearson"):
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if not num_cols:
        return pd.DataFrame()
    return df[num_cols].corr(method=method)
