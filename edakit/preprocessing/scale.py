
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def standard_scale(df: pd.DataFrame, cols=None):
    out = df.copy()
    cols = cols or [c for c in out.columns if is_numeric_dtype(out[c])]
    for c in cols:
        s = out[c].astype(float)
        mu = s.mean(); sd = s.std()
        if sd == 0 or pd.isna(sd):
            continue
        out[c] = (s - mu) / sd
    return out

def robust_scale(df: pd.DataFrame, cols=None):
    out = df.copy()
    cols = cols or [c for c in out.columns if is_numeric_dtype(out[c])]
    for c in cols:
        s = out[c].astype(float)
        med = s.median(); q1 = s.quantile(0.25); q3 = s.quantile(0.75)
        iqr = q3 - q1 if pd.notna(q3) and pd.notna(q1) else None
        if iqr in (None, 0):
            continue
        out[c] = (s - med) / iqr
    return out
