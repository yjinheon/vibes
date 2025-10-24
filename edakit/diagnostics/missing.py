
import pandas as pd

def missing_matrix(df: pd.DataFrame):
    miss = df.isna()
    cols = miss.columns.tolist()
    patterns = miss.groupby(cols, dropna=False).size().reset_index(name="rows")
    return patterns.sort_values("rows", ascending=False)
