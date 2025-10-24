
import numpy as np
import pandas as pd

def entropy_from_counts(counts):
    p = counts[counts>0]/counts.sum()
    return float(-(p*np.log2(p)).sum())

def text_summary(df: pd.DataFrame):
    texts = [c for c in df.columns if df[c].dtype == object]
    if not texts:
        return pd.DataFrame()
    rows = []
    for c in texts:
        s = df[c].astype("string")
        lens = s.str.len()
        words = s.str.split().map(lambda x: len(x) if isinstance(x, list) else float("nan"))
        rows.append({
            "variable": c,
            "missing_ratio": s.isna().mean()*100,
            "avg_len": float(np.nanmean(lens)),
            "p95_len": float(np.nanpercentile(lens.dropna(), 95)) if lens.notna().any() else float("nan"),
            "avg_words": float(np.nanmean(words)),
            "entropy": entropy_from_counts(s.value_counts(dropna=True).values) if s.notna().any() else float("nan")
        })
    return pd.DataFrame(rows)
