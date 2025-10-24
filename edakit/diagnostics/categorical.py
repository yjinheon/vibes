import pandas as pd
from pandas.api.types import is_categorical_dtype


def categorical_summary(df: pd.DataFrame, max_levels: int = 20):
    cats = [
        c for c in df.columns if is_categorical_dtype(df[c]) or df[c].dtype == object
    ]
    if not cats:
        return pd.DataFrame()

    # Handle empty DataFrame case
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["variable", "level", "freq", "ratio"])

    frames = []
    for c in cats:
        s = df[c]
        vc = s.value_counts(dropna=False).head(max_levels)

        # Skip if no values found
        if len(vc) == 0:
            continue

        # Handle pandas version compatibility for reset_index column naming
        d = vc.rename("freq").to_frame().reset_index()
        # The index column might be named 'index' (older pandas) or the column name (newer pandas)
        index_col = d.columns[0] if d.columns[0] != "freq" else "index"
        d = d.rename(columns={index_col: "level"})

        d["variable"] = c
        d["ratio"] = d["freq"] / n * 100
        frames.append(d[["variable", "level", "freq", "ratio"]])

    # Handle case where no valid categorical data was found
    if not frames:
        return pd.DataFrame(columns=["variable", "level", "freq", "ratio"])

    return pd.concat(frames, axis=0).reset_index(drop=True)
