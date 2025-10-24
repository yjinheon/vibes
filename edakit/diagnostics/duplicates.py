
import pandas as pd

def duplicates_overview(df: pd.DataFrame, subset=None):
    mask = df.duplicated(subset=subset, keep=False)
    return pd.DataFrame([{
        "duplicated_rows_cnt": int(mask.sum()),
        "duplicated_rows_ratio": float(mask.mean()*100)
    }])
