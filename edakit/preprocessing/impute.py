
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def simple_impute(df: pd.DataFrame, num_strategy: str = "median", cat_fill: str = "Missing"):
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if is_numeric_dtype(s):
            if num_strategy == "mean":
                val = float(s.mean())
            elif num_strategy == "median":
                val = float(s.median())
            else:
                val = float(s.mode(dropna=True).iloc[0]) if s.dropna().size>0 else 0.0
            out[c] = s.fillna(val)
        else:
            out[c] = s.fillna(cat_fill)
    return out
