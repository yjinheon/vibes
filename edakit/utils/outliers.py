
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from .types import nanpercentile

def iqr_bounds(s: pd.Series, factor: float = 1.5):
    x = pd.to_numeric(s, errors="coerce").to_numpy()
    q1, q3 = nanpercentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr

def zscore_mask(s: pd.Series, thr: float = 3.0):
    x = pd.to_numeric(s, errors="coerce").to_numpy().astype(float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x, dtype=bool)
    z = (x - mu) / sd
    return np.abs(z) > thr

def mad_mask(s: pd.Series, thr: float = 3.5):
    x = pd.to_numeric(s, errors="coerce").to_numpy().astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x, dtype=bool)
    z = 0.6745 * (x - med) / mad
    return np.abs(z) > thr

def outlier_mask(s: pd.Series, method: str, iqr_factor: float, z_thr: float, mad_thr: float):
    if not is_numeric_dtype(s):
        return np.zeros(len(s), dtype=bool)
    if method == "iqr":
        lo, hi = iqr_bounds(s, iqr_factor)
        x = pd.to_numeric(s, errors="coerce")
        return (x < lo) | (x > hi)
    if method == "zscore":
        return zscore_mask(s, z_thr)
    if method == "mad":
        return mad_mask(s, mad_thr)
    raise ValueError(f"Unknown outlier method: {method}")
