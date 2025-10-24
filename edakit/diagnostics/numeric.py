import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from ..utils.outliers import outlier_mask, iqr_bounds


def numeric_summary(
    df: pd.DataFrame, method: str, iqr_factor: float, z_thr: float, mad_thr: float
):

    num_cols = df.select_dtypes(include=["int", "float"]).columns

    if len(num_cols) == 0:
        return pd.DataFrame()

    desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skew"] = df[num_cols].skew()
    desc["kurt"] = df[num_cols].kurt()

    out_cnt, out_ratio, lo_b, hi_b, with_mean, wo_mean = {}, {}, {}, {}, {}, {}

    for c in num_cols:
        s = df[c]

        if not is_numeric_dtype(s) or s.dtype == bool:
            continue

        m = outlier_mask(s, method, iqr_factor, z_thr, mad_thr)

        try:
            lo, hi = iqr_bounds(s, iqr_factor)
        except TypeError:
            lo, hi = np.nan, np.nan

        out_cnt[c] = int(m.sum())
        out_ratio[c] = float(m.mean() * 100)
        lo_b[c], hi_b[c] = lo, hi
        with_mean[c] = float(np.nanmean(s.values)) if s.notna().any() else np.nan
        wo_mean[c] = float(np.nanmean(s[~m].values)) if (~m).any() else np.nan

    desc["outliers_cnt"] = pd.Series(out_cnt)
    desc["outliers_ratio"] = pd.Series(out_ratio)
    desc["iqr_lower"] = pd.Series(lo_b)
    desc["iqr_upper"] = pd.Series(hi_b)
    desc["mean_with_outliers"] = pd.Series(with_mean)
    desc["mean_without_outliers"] = pd.Series(wo_mean)

    return desc
