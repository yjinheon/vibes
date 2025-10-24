
import pandas as pd
import numpy as np

# ----- Calendar features -----
def add_calendar_parts(df: pd.DataFrame, dt_col: str, prefix: str = None):
    prefix = prefix or dt_col
    s = pd.to_datetime(df[dt_col], errors="coerce")
    out = df.copy()
    out[f"{prefix}_year"] = s.dt.year
    out[f"{prefix}_quarter"] = s.dt.quarter
    out[f"{prefix}_month"] = s.dt.month
    out[f"{prefix}_week"] = s.dt.isocalendar().week.astype("Int64")
    out[f"{prefix}_day"] = s.dt.day
    out[f"{prefix}_weekday"] = s.dt.weekday  # Monday=0
    out[f"{prefix}_is_weekend"] = out[f"{prefix}_weekday"].isin([5,6]).astype("Int8")
    out[f"{prefix}_hour"] = s.dt.hour
    out[f"{prefix}_minute"] = s.dt.minute
    out[f"{prefix}_second"] = s.dt.second
    return out

# ----- Cyclical encoding -----
def add_cyclical(df: pd.DataFrame, col: str, max_val: int, prefix: str = None):
    prefix = prefix or col
    out = df.copy()
    out[f"{prefix}_sin"] = np.sin(2*np.pi * df[col] / max_val)
    out[f"{prefix}_cos"] = np.cos(2*np.pi * df[col] / max_val)
    return out

# ----- Elapsed time since/between -----
def add_elapsed_since(df: pd.DataFrame, dt_col: str, ref: pd.Timestamp = None, unit: str = "days", prefix: str = None):
    prefix = prefix or dt_col
    s = pd.to_datetime(df[dt_col], errors="coerce")
    ref = ref or s.min()
    delta = (s - ref)
    conv = {"days":1, "hours":24, "minutes":24*60, "seconds":24*3600}
    val = delta.dt.total_seconds() / (24*3600)  # days
    if unit != "days":
        val = val * conv[unit]
    out = df.copy()
    out[f"{prefix}_elapsed_{unit}"] = val
    return out

def add_diff_between(df: pd.DataFrame, dt_col_a: str, dt_col_b: str, unit: str = "days", prefix: str = None):
    prefix = prefix or f"{dt_col_b}_minus_{dt_col_a}"
    a = pd.to_datetime(df[dt_col_a], errors="coerce")
    b = pd.to_datetime(df[dt_col_b], errors="coerce")
    delta = (b - a).dt.total_seconds()
    conv = {"days": 86400, "hours": 3600, "minutes": 60, "seconds": 1}
    out = df.copy()
    out[f"{prefix}_{unit}"] = delta / conv[unit]
    return out

# ----- Rolling/Window features per key -----
def add_rolling_stats(df: pd.DataFrame, key: str, dt_col: str, value_col: str, windows=(3,7,30)):
    out = df.copy().sort_values([key, dt_col])
    for w in windows:
        out[f"{value_col}_rollmean_{w}"] = out.groupby(key)[value_col].transform(lambda s: s.rolling(w, min_periods=1).mean())
        out[f"{value_col}_rollstd_{w}"]  = out.groupby(key)[value_col].transform(lambda s: s.rolling(w, min_periods=1).std())
        out[f"{value_col}_rollmin_{w}"]  = out.groupby(key)[value_col].transform(lambda s: s.rolling(w, min_periods=1).min())
        out[f"{value_col}_rollmax_{w}"]  = out.groupby(key)[value_col].transform(lambda s: s.rolling(w, min_periods=1).max())
    return out

# ----- Lag/Lead -----
def add_lags(df: pd.DataFrame, key: str, dt_col: str, value_col: str, lags=(1,2,3)):
    out = df.copy().sort_values([key, dt_col])
    for l in lags:
        out[f"{value_col}_lag_{l}"] = out.groupby(key)[value_col].shift(l)
    return out

def add_leads(df: pd.DataFrame, key: str, dt_col: str, value_col: str, leads=(1,2,3)):
    out = df.copy().sort_values([key, dt_col])
    for l in leads:
        out[f"{value_col}_lead_{l}"] = out.groupby(key)[value_col].shift(-l)
    return out

# ----- Sessionization -----
def add_sessions(df: pd.DataFrame, key: str, dt_col: str, gap_minutes: int = 30, session_col: str = "session_id"):
    out = df.copy().sort_values([key, dt_col])
    t = pd.to_datetime(out[dt_col], errors="coerce")
    delta = t.diff().dt.total_seconds().fillna(0) / 60.0
    new_session = (delta > gap_minutes).astype(int)
    out[session_col] = new_session.cumsum()
    # Make sessions per key
    out[session_col] = out.groupby(key)[session_col].transform(lambda s: (s.diff().fillna(1)>0).cumsum())
    return out

def session_stats(df: pd.DataFrame, key: str, session_col: str, value_col: str = None):
    g = df.groupby([key, session_col])
    res = g.size().rename("events_per_session").to_frame()
    res["session_duration_min"] = g["timestamp"].apply(lambda s: (pd.to_datetime(s).max() - pd.to_datetime(s).min()).total_seconds()/60.0 if s.size>1 else 0.0)
    if value_col is not None and value_col in df.columns:
        res[f"{value_col}_sum_per_session"] = g[value_col].sum()
        res[f"{value_col}_mean_per_session"] = g[value_col].mean()
    return res.reset_index()
