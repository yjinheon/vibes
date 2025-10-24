
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def _discretize(s: pd.Series, bins=10):
    try:
        return pd.qcut(s, q=bins, duplicates="drop")
    except Exception:
        return pd.cut(s, bins=bins)

def mutual_info_approx(df: pd.DataFrame, target: str, bins: int = 10):
    # 간단 MI 근사: 범주화 후 엔트로피 기반
    if target not in df.columns:
        return pd.DataFrame()
    t = df[target]
    rows = []
    for c in df.columns:
        if c == target: continue
        x = df[c]
        if is_numeric_dtype(x):
            x = _discretize(x, bins=bins)
        joint = pd.crosstab(x, t)
        pxy = joint.values / joint.values.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(pxy * (np.log(pxy+1e-12) - np.log(px+1e-12) - np.log(py+1e-12)))
        rows.append({"feature": c, "mi": float(mi)})
    return pd.DataFrame(rows).sort_values("mi", ascending=False)
