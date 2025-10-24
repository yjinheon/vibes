
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

def mem_usage_mb(s: pd.Series) -> float:
    return round(s.memory_usage(deep=True) / (1024**2), 6)

def nanpercentile(a, q):
    return np.nanpercentile(a, q)

def ensure_datetime_cols(df: pd.DataFrame, parse: bool = True) -> pd.DataFrame:
    if not parse:
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            try:
                parsed = pd.to_datetime(out[c], errors="raise", infer_datetime_format=True)
                if parsed.notna().mean() >= 0.8:
                    out[c] = pd.to_datetime(out[c], errors="coerce", infer_datetime_format=True)
            except Exception:
                pass
    return out

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if is_numeric_dtype(s):
            if "float" in str(s.dtype):
                out[c] = pd.to_numeric(s, downcast="float")
            elif "int" in str(s.dtype):
                out[c] = pd.to_numeric(s, downcast="integer")
    return out
