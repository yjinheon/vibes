
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    n = ct.values.sum()
    if n == 0: return float("nan")
    expected = np.outer(ct.sum(axis=1).values, ct.sum(axis=0).values)/n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((ct.values - expected)**2 / expected)
    r, k = ct.shape
    denom = n*(min(r-1, k-1))
    if denom <= 0: return float("nan")
    return float(np.sqrt(chi2/denom))

def association_categorical(df: pd.DataFrame):
    cats = [c for c in df.columns if is_categorical_dtype(df[c]) or df[c].dtype == object]
    cats = [c for c in cats if df[c].nunique(dropna=True) > 1]
    if len(cats) < 2:
        return pd.DataFrame()
    mat = pd.DataFrame(np.nan, index=cats, columns=cats)
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if j < i: continue
            if a == b:
                mat.loc[a, b] = 1.0
            else:
                v = cramers_v(df[a], df[b])
                mat.loc[a, b] = v
                mat.loc[b, a] = v
    return mat
