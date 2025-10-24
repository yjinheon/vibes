
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def low_variance_columns(df: pd.DataFrame, threshold: float = 0.0):
    rows = []
    for c in df.columns:
        s = df[c]
        if is_numeric_dtype(s):
            var = float(np.nanvar(s.values))
            if var <= threshold:
                rows.append({"variable": c, "variance": var})
        else:
            nunique = s.nunique(dropna=True)
            if nunique <= 1:
                rows.append({"variable": c, "variance": 0.0})
    return pd.DataFrame(rows)
