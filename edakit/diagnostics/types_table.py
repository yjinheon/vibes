
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from ..utils.types import mem_usage_mb

def types_table(df: pd.DataFrame):
    dtypes = df.dtypes.to_frame("dtype")
    miss_cnt = df.isna().sum().to_frame("missing_count")
    miss_ratio = (miss_cnt["missing_count"]/len(df)*100).to_frame("missing_ratio")
    nunique = df.nunique(dropna=True).to_frame("unique_count")
    uniq_ratio = (nunique["unique_count"]/len(df)*100).to_frame("unique_ratio")
    mem = df.apply(mem_usage_mb).to_frame("memory_mb")

    zeros, negatives, mins, maxs = {}, {}, {}, {}
    for c in df.columns:
        s = df[c]
        if is_numeric_dtype(s):
            zeros[c] = int((s==0).sum())
            negatives[c] = int((s<0).sum())
            mins[c] = float(np.nanmin(s.values)) if s.notna().any() else np.nan
            maxs[c] = float(np.nanmax(s.values)) if s.notna().any() else np.nan
        else:
            zeros[c] = negatives[c] = mins[c] = maxs[c] = np.nan

    tbl = dtypes.join([miss_cnt, miss_ratio, nunique, uniq_ratio, mem])\
                 .assign(zeros_cnt=pd.Series(zeros), negative_cnt=pd.Series(negatives),
                         min=pd.Series(mins), max=pd.Series(maxs))
    return tbl
