
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def psi_numeric(base: pd.Series, cur: pd.Series, bins=10):
    # Percentile binning
    qs = np.linspace(0, 1, bins+1)
    edges = np.unique(np.nanpercentile(base.dropna(), qs*100))
    if len(edges) < 3:
        return float("nan")
    base_hist, _ = np.histogram(base.dropna(), bins=edges)
    cur_hist, _  = np.histogram(cur.dropna(),  bins=edges)
    base_p = base_hist / max(base_hist.sum(), 1)
    cur_p  = cur_hist  / max(cur_hist.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.nansum((cur_p - base_p) * np.log((cur_p + 1e-12) / (base_p + 1e-12)))
    return float(psi)

def psi_frame(base_df: pd.DataFrame, cur_df: pd.DataFrame, cols=None):
    cols = cols or [c for c in base_df.columns if is_numeric_dtype(base_df[c])]
    rows = []
    for c in cols:
        rows.append({"feature": c, "psi": psi_numeric(base_df[c], cur_df[c])})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
