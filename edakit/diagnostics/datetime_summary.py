
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

def datetime_summary(df: pd.DataFrame):
    dts = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    if not dts:
        return pd.DataFrame()
    rows = []
    for c in dts:
        s = df[c]
        rows.append({
            "variable": c,
            "min": s.min(),
            "max": s.max(),
            "missing_count": int(s.isna().sum()),
            "missing_ratio": s.isna().mean()*100,
            "range_days": (s.max()-s.min()).days if s.notna().any() else float("nan")
        })
    return pd.DataFrame(rows)
