
import pandas as pd
from pandas.api.types import is_numeric_dtype
from ..utils.outliers import outlier_mask, iqr_bounds

def outliers_table(df: pd.DataFrame, method: str, iqr_factor: float, z_thr: float, mad_thr: float):
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    rows = []
    for c in num_cols:
        s = df[c]
        m = outlier_mask(s, method, iqr_factor, z_thr, mad_thr)
        lo, hi = iqr_bounds(s, iqr_factor)
        rows.append({
            "variable": c,
            "outliers_cnt": int(m.sum()),
            "outliers_ratio": float(m.mean()*100),
            "iqr_lower": lo,
            "iqr_upper": hi,
        })
    return pd.DataFrame(rows).sort_values("outliers_ratio", ascending=False)
